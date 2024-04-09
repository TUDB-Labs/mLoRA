import mlora
import torch
import json
import fire


def main(base_model: str,
         task_name: str,
         lora_weights: str = None,
         load_16bit: bool = True,
         load_8bit: bool = False,
         load_4bit: bool = False,
         flash_attn: bool = False,
         save_file: str = None,
         batch_size: int = 32,
         router_profile: bool = False,
         device: str = f"{mlora.get_backend().device_name()}:0"):

    mlora.setup_logging("INFO")

    if not mlora.get_backend().check_available():
        exit(-1)

    model = mlora.LLMModel.from_pretrained(base_model, device=device,
                                           attn_impl="flash_attn" if flash_attn else "eager",
                                           bits=(8 if load_8bit else (
                                                 4 if load_4bit else None)),
                                           load_dtype=torch.bfloat16 if load_16bit else torch.float32)
    tokenizer = mlora.Tokenizer(base_model)
    adapter_name = model.load_adapter_weight(
        lora_weights if lora_weights else "default")
    evaluate_paramas = mlora.EvaluateConfig(
        adapter_name=adapter_name,
        task_name=task_name,
        batch_size=batch_size,
        router_profile=router_profile)

    output = mlora.evaluate(
        model, tokenizer, [evaluate_paramas], save_file=save_file)

    print(json.dumps(output, indent=4))


if __name__ == "__main__":
    fire.Fire(main)
