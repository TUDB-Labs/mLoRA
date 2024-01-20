import mlora
import fire


def main(base_model: str,
         instruction: str,
         input: str = None,
         template: str = None,
         lora_weights: str = None,
         load_8bit: bool = False,
         load_4bit: bool = False,
         device: str = "cuda:0"):

    model = mlora.LlamaModel.from_pretrained(base_model, device=device,
                                             bits=(8 if load_8bit else (4 if load_4bit else None)))
    tokenizer = mlora.Tokenizer(base_model)

    if lora_weights:
        adapter_name = model.load_adapter_weight(lora_weights)
        generate_paramas = model.get_generate_paramas()[adapter_name]
    else:
        adapter_name = base_model
        generate_paramas = mlora.GenerateConfig(adapter_name_=adapter_name)

    generate_paramas.prompt_template_ = template
    generate_paramas.prompts_ = [(instruction, input)]

    output = mlora.generate(model, tokenizer, [generate_paramas],
                            temperature=0.5, top_p=0.9, max_gen_len=128,
                            device=device)

    for prompt in output[adapter_name]:
        print(f"\n{'='*10}\n")
        print(prompt)
        print(f"\n{'='*10}\n")


if __name__ == "__main__":
    fire.Fire(main)
