import mlora
import torch
import fire


def inference_callback(cur_pos, outputs):
    print(f"Position: {cur_pos}")
    for adapter_name, output in outputs.items():
        print(f"{adapter_name} output: {output[0]}")


def main(base_model: str,
         instruction: str,
         input: str = None,
         template: str = None,
         lora_weights: str = None,
         load_16bit: bool = True,
         load_8bit: bool = False,
         load_4bit: bool = False,
         flash_attn: bool = False,
         max_seq_len: int = None,
         stream: bool = False,
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
        prompts=[(instruction, input)])

    if max_seq_len is None:
        max_seq_len = model.config_.max_seq_len_

    output = mlora.generate(
        model, tokenizer, [generate_paramas], max_gen_len=max_seq_len, stream_callback=inference_callback if stream else None)

    for prompt in output[adapter_name]:
        print(f"\n{'='*10}\n")
        print(prompt)
        print(f"\n{'='*10}\n")


if __name__ == "__main__":
    fire.Fire(main)
