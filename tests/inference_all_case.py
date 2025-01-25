import torch
import argparse
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

G_TEST_ADAPTERS = [
    # lora adapter
    "adapters/lora_sft_0",
    "adapters/lora_sft_1",
    # loraplus adapter
    "adapters/loraplus_sft_0",
    "adapters/loraplus_sft_1",
    # dpo adapter
    "adapters/lora_base_dpo",
    "adapters/lora_sft_dpo",
    "adapters/loraplus_sft_dpo",
    # cpo adapter
    "adapters/lora_cpo",
    "adapters/loraplus_cpo"
    # ppo adapter
    "adapters/lora_ppo_actor"
]


def get_cmd_args():
    parser = argparse.ArgumentParser(description='mLoRA test function')
    parser.add_argument('--base_model', type=str, required=True,
                        help='Path to or name of base model')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_cmd_args()

    model_path = args.base_model

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # load adapter
    for adapter in G_TEST_ADAPTERS:
        model.load_adapter(adapter, adapter_name=adapter)
    model.disable_adapters()

    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model.enable_adapters()

    for adapter in G_TEST_ADAPTERS:
        model.set_adapter(adapter)

        sequences = pipeline(
            "What is mLoRA?",
            do_sample=True,
            top_k=1,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            max_length=128,
            truncation=True,
        )

        print(sequences)
        output: str = sequences[0]['generated_text'].strip().replace(
            "\r", " ").replace("\n", " ")
        print(f"Adapter {adapter} Output is: {output}")
