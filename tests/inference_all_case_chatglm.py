import torch
import argparse
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

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
    # cit adapter
    "adapters/lora_cit",
    "adapters/loraplus_cit"
]


def get_cmd_args():
    parser = argparse.ArgumentParser(description='mLoRA test function')
    parser.add_argument('--base_model', type=str, required=True,
                        help='Path to or name of base model')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_cmd_args()
    
    model_path = args.base_model

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    
    query = "What is mLoRA?"
    device = "cuda"

    inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                       add_generation_prompt=True,
                                       tokenize=True,
                                       return_tensors="pt",
                                       return_dict=True
                                       )
    inputs = inputs.to(device)

    model.to(device).eval()
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    
    for adapter in G_TEST_ADAPTERS:
        
        peft_model = PeftModel.from_pretrained(model, adapter)


        with torch.no_grad():
            #print(peft_model.generate(**inputs))
            outputs = peft_model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            print(tokenizer.decode(outputs[0], skip_special_tokens=True))