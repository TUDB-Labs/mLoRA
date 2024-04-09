import mlora
import torch
import fire


def main(base_model: str,
         instruction: str,
         input: str = None,
         template: str = None,
         lora_weights: str = None,
         load_16bit: bool = True,
         load_8bit: bool = False,
         load_4bit: bool = False,
         flash_attn: bool = False,
         device: str = f"{mlora.get_backend().device_name()}:0"):

    model = mlora.LLMModel.from_pretrained(base_model, device=device,
                                           attn_impl="flash_attn" if flash_attn else "eager",
                                           bits=(8 if load_8bit else (
                                                 4 if load_4bit else None)),
                                           load_dtype=torch.bfloat16 if load_16bit else torch.float32)
    tokenizer = mlora.Tokenizer(base_model)
    adapter_name = model.load_adapter_weight(
        lora_weights if lora_weights else "default")
    generate_paramas = mlora.GenerateConfig(
        adapter_name=adapter_name,
        prompt_template=template,
        prompts=[(instruction, input)],
        temperature=0.5, top_p=0.9)

    output = mlora.generate(
        model, tokenizer, [generate_paramas], max_gen_len=128)

    for prompt in output[adapter_name]:
        print(f"\n{'='*10}\n")
        print(prompt)
        print(f"\n{'='*10}\n")


if __name__ == "__main__":
    fire.Fire(main)
