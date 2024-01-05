import argparse
import mlora

# Command Line Arguments
parser = argparse.ArgumentParser(description='m-LoRA LLM generator')
parser.add_argument('--base_model', type=str, required=True,
                    help='Path to or name of base model')
parser.add_argument('--instruction', type=str, required=True,
                    help='Instruction of the prompt')
parser.add_argument('--input', type=str,
                    help='Input of the prompt')
parser.add_argument('--template', type=str,
                    help='Prompt template')
parser.add_argument('--load_8bit', action="store_true",
                    help='Load model in 8bit mode')
parser.add_argument('--load_4bit', action="store_true",
                    help='Load model in 4bit mode')
parser.add_argument('--lora_weights', type=str,
                    help='Path to or name of LoRA weights')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='Specify which GPU to be used, default is cuda:0')
args = parser.parse_args()

model = mlora.LlamaModel.from_pretrained(args.base_model, device=args.device,
                                         bits=(8 if args.load_8bit else (4 if args.load_4bit else None)))
tokenizer = mlora.Tokenizer(args.base_model, device=args.device)

if args.lora_weights:
    adapter_name = model.load_adapter_weight(args.lora_weights)
    generate_paramas = model.get_generate_paramas()[adapter_name]
else:
    adapter_name = args.base_model
    generate_paramas = mlora.GenerateConfig(adapter_name_=adapter_name)

generate_paramas.prompt_template_ = args.template
generate_paramas.prompts_ = [
    generate_paramas.generate_prompt(args.instruction, args.input)]

output = mlora.generate(model, tokenizer, [generate_paramas],
                        temperature=0.5, top_p=0.9, max_gen_len=128,
                        device=args.device)

for prompt in output[adapter_name]:
    print(f"\n{'='*10}\n")
    print(prompt)
    print(f"\n{'='*10}\n")
