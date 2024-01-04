import mlora

model_name = "decapoda-research/llama-7b-hf"
lora_name = "tloen/alpaca-lora-7b"
target_device = "cuda:0"
prompt_template = "template/alpaca.json"
prompts = ["Tell me about China."]

model = mlora.LlamaModel.from_pretrained(
    model_name, device=target_device, bits=8)
tokenizer = mlora.Tokenizer(model_name, device=target_device)
model.pad_token_id_ = tokenizer.pad_id_

adapter_name = model.load_adapter_weight(lora_name)

output = mlora.generate(model, tokenizer,
                        model.get_generate_paramas(prompts),
                        temperature=0.5, top_p=0.9, max_gen_len=128,
                        device=target_device)

for prompt in output[adapter_name]:
    print(prompt)
