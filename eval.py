import json
import torch
import argparse
from peft import PeftModel
from aspen.evaluator import Evaluator
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, BitsAndBytesConfig

parser = argparse.ArgumentParser()
parser.add_argument('--no_lora', type=bool, default=False)
parser.add_argument('--load_8bit', type=bool, default=False)
parser.add_argument('--load_4bit', type=bool, default=False)
parser.add_argument('--base_model', type=str)
parser.add_argument('--lora_weights', type=str)  # lora checkpoint
parser.add_argument('--model_type', default="llama", choices=['llama', 'chatglm', 'bloom'])
parser.add_argument('--dataset', type=str)
parser.add_argument('--device', type=str, default="cuda:0")
args = parser.parse_args()

if args.base_model is None:
    print('error: Argument --base_model are required.')
    parser.print_help()
    exit(-1)

if args.dataset is None:
    print('error: Argument --dataset are required.')
    parser.print_help()
    exit(-1)

if args.no_lora is False:
    if args.lora_weights is None:
        print('error: Argument --lora_weights are required.')
        parser.print_help()
        exit(-1)

if args.load_8bit and args.load_4bit:
    print('error: only one of --load_4bit and --load_8bit can be true.')
    parser.print_help()
    exit(-1)

print(args)

NO_LORA = args.no_lora
LOAD_8BIT = args.load_8bit
LOAD_4BIT = args.load_4bit
BASE_MODEL = args.base_model
LORA_WEIGHTS = args.lora_weights
DATASET = args.dataset

if args.model_type == "llama":
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL, legacy=False)
else:
    raise "currently only the llama model is supported."

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if device == "cuda":
    device = args.device
    compute_dtype = (torch.float16 if LOAD_8BIT else (torch.bfloat16 if LOAD_4BIT else torch.float32))
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL,
        load_in_8bit=LOAD_8BIT,
        load_in_4bit=LOAD_4BIT,
        device_map=args.device,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=LOAD_4BIT,
            load_in_8bit=LOAD_8BIT,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    )
    if not NO_LORA:
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            torch_dtype=torch.float16,
            device_map=args.device,
        )

else:  # cpu
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, device_map={"": device}, low_cpu_mem_usage=True
    )
    if not NO_LORA:
        model = PeftModel.from_pretrained(
            model,
            LORA_WEIGHTS,
            device_map={"": device},
        )

evaluator = Evaluator()
model.eval()


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context.
         Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:"""
    else:
        return f"""Below is an instruction that describes a task.
         Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:"""


def get_scores(data_path: str, temperature=1.0, top_p=0.9, top_k=40, num_beams=4, max_new_tokens=1024,
               **kwargs) -> dict:
    with open(data_path, 'r', encoding='utf8') as fp:
        dataset = json.load(fp)
        datasize = len(dataset)
        scores = {'rouge-1': 0, 'rouge-2': 0, 'rouge-l': 0, 'bleu-2': 0}
        for item_data in dataset:
            inst = item_data['instruction']
            data_in = item_data['input']
            true_out = item_data['output']
            gen_out = evaluate(inst, data_in)

            score_dict = evaluator.calculate_ROUGE(gen_out, true_out)
            scores['rouge-1'] = scores['rouge-1'] + score_dict['rouge-1']
            scores['rouge-2'] = scores['rouge-2'] + score_dict['rouge-2']
            scores['rouge-l'] = scores['rouge-l'] + score_dict['rouge-l']
            scores['bleu-2'] = evaluator.calculate_BLEU(gen_out, true_out, 2)['bleu-2']

        scores['rouge-1'] = round(scores['rouge-1'] / datasize, 2)
        scores['rouge-2'] = round(scores['rouge-2'] / datasize, 2)
        scores['rouge-l'] = round(scores['rouge-l'] / datasize, 2)
        scores['bleu-2'] = round(scores['bleu-2'] / datasize, 2)
        return scores


def evaluate(instruction, input=None, max_new_tokens=1024, ):
    prompt = generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generate_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
    output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)[0]
    output = output.split("### Response:")[1].strip()
    return output


if __name__ == '__main__':
    print(get_scores(DATASET))
