from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
from peft import PeftModel
import sys


def inference_with_finetuned_lora(base_model_name_or_path, lora_weights_path, prompt):
    tokenizer = LlamaTokenizer.from_pretrained(base_model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(
        base_model_name_or_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights_path,
        torch_dtype=torch.float16,
    )

    generation_config = model.generation_config

    device = "cuda:0"
    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config
        )

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_text.replace(prompt, '').strip()
        print(answer)

if __name__ == '__main__':
    args = sys.argv
    method = args[1]
    if method == 'inference':
        base_model_or_path = args[2]
        lora_weight_path = args[3]
        prompt = args[4]
        inference_with_finetuned_lora(base_model_or_path, lora_weight_path, prompt)