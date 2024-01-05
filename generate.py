import mlora

model_name = "yahma/llama-7b-hf"
lora_name = "tloen/alpaca-lora-7b"
target_device = "cuda:0"

model = mlora.LlamaModel.from_pretrained(
    model_name, device=target_device, bits=8)
tokenizer = mlora.Tokenizer(model_name, device=target_device)

adapter_name = model.load_adapter_weight(lora_name)
generate_paramas = model.get_generate_paramas()[adapter_name]
generate_paramas.prompts_ = ["Tell me about China."]
generate_paramas.prompt_template_ = "template/alpaca.json"

output = mlora.generate(model, tokenizer, [generate_paramas],
                        temperature=0.5, top_p=0.9, max_gen_len=128,
                        device=target_device)

for prompt in output[adapter_name]:
    print(prompt)
