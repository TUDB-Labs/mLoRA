import fire
import torch

import mlora


def main(
    base_model: str,
    adapter_name: str = "lora_0",
    train_data: str = "TUDB-Labs/Dummy-mLoRA",
    test_prompt: str = "Could you provide an introduction to m-LoRA?",
):
    mlora.setup_logging("INFO")

    model: mlora.LLMModel = mlora.LLMModel.from_pretrained(
        base_model,
        device=mlora.get_backend().default_device_name(),
        load_dtype=torch.bfloat16,
    )
    tokenizer = mlora.Tokenizer(base_model)

    lora_config = mlora.LoraConfig(
        adapter_name=adapter_name,
        lora_r_=32,
        lora_alpha_=64,
        lora_dropout_=0.05,
        target_modules_={
            "q_proj": True,
            "k_proj": True,
            "v_proj": True,
            "o_proj": True,
        },
    )

    model.init_adapter(lora_config)

    train_config = mlora.TrainConfig(
        num_epochs=10,
        batch_size=16,
        micro_batch_size=8,
        learning_rate=1e-4,
        casual_train_data=train_data,
    ).init_for(lora_config)

    mlora.train(model=model, tokenizer=tokenizer, configs=[train_config])

    lora_config, lora_weight = model.unload_adapter(adapter_name)

    model.init_adapter(lora_config, lora_weight)
    model.init_adapter(mlora.AdapterConfig(adapter_name="default"))

    generate_configs = [
        mlora.GenerateConfig(
            adapter_name=adapter_name,
            prompts=[test_prompt],
            stop_token="\n",
        ),
        mlora.GenerateConfig(
            adapter_name="default",
            prompts=[test_prompt],
            stop_token="\n",
        ),
    ]

    outputs = mlora.generate(
        model=model,
        tokenizer=tokenizer,
        configs=generate_configs,
        max_gen_len=128,
    )

    print(f"\n{'='*10}\n")
    print(f"PROMPT: {test_prompt}")
    for adapter_name, output in outputs.items():
        print(f"{adapter_name} OUTPUT:")
        print(output[0])
    print(f"\n{'='*10}\n")


if __name__ == "__main__":
    fire.Fire(main)
