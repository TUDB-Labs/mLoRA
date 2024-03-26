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
         save_file: str = None,
         device: str = "cuda:0"):

    model = mlora.LlamaModel.from_pretrained(base_model, device=device,
                                             bits=(8 if load_8bit else (
                                                 4 if load_4bit else None)),
                                             load_dtype=torch.bfloat16 if load_16bit else torch.float32)
    tokenizer = mlora.Tokenizer(base_model)
    adapter_name = model.load_adapter_weight(
        lora_weights if lora_weights else "default")
    evaluate_paramas = mlora.EvaluateConfig(
        adapter_name_=adapter_name,
        task_name_=task_name,
        batch_size_=32)

    output = mlora.evaluate(
        model, tokenizer, [evaluate_paramas], save_file=save_file)

    print(json.dumps(output, indent=4))


if __name__ == "__main__":
    fire.Fire(main)
