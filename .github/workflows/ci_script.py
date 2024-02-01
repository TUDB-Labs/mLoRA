from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel
import torch
from peft import PeftModel
import sys


def inference_llama(base_model_name_or_path, lora_weights_path, prompt, device):
    tokenizer = LlamaTokenizer.from_pretrained(base_model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.float16,
        device_map=device,
    )

    return inference(tokenizer, model, device, prompt, lora_weights_path)


def inference_chatglm(base_model_name_or_path, lora_weights_path, prompt, device):
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        base_model_name_or_path, trust_remote_code=True).to(device)

    return inference(tokenizer, model, device, prompt, lora_weights_path)


def inference(tokenizer, model, device, prompt, lora_weights_path):
    model = PeftModel.from_pretrained(
        model, lora_weights_path, torch_dtype=torch.float16)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    output = ""
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=100, do_sample=True, top_p=1, temperature=1
        )
        output = tokenizer.batch_decode(outputs.detach().cpu(
        ).numpy(), skip_special_tokens=True)[0][len(prompt):]

        print(f"Prompt:\n{prompt}\n")
        print(f"Generated:\n{output}")

    return output


if __name__ == '__main__':
    args = sys.argv
    model_type = args[1]
    base_model_or_path = args[2]
    lora_weight_path = args[3]
    prompt = args[4]

    check_prompt = None
    if len(args) >= 6:
        check_prompt = args[5]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_type == 'llama':
        output = inference_llama(
            base_model_or_path, lora_weight_path, prompt, device)
    elif model_type == 'chatglm':
        output = inference_chatglm(
            base_model_or_path, lora_weight_path, prompt, device)
    else:
        print(f"no such model type: {model_type}.")
        exit(1)

    if check_prompt is not None and check_prompt not in output:
        exit(1)

    exit(0)
