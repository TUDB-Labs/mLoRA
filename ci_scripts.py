from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel
import torch
from peft import PeftModel
import sys


def inference_llama(base_model_name_or_path, lora_weights_path, prompt):
    device = "cuda:0"
    tokenizer = LlamaTokenizer.from_pretrained(base_model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(
        base_model_name_or_path,
        load_in_8bit=True,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model = PeftModel.from_pretrained(
        model,
        lora_weights_path,
        torch_dtype=torch.float16,
    )
    inference(tokenizer, model, device)


def inference_chatglm2(base_model_name_or_path, lora_weights_path, prompt):
    device = "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(base_model_name_or_path, trust_remote_code=True).to(device)
    model = PeftModel.from_pretrained(model, lora_weights_path)
    inference(tokenizer, model, device)


def inference(tokenizer, model, device):
    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    generation_config = model.generation_config
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
    base_model_or_path = args[2]
    lora_weight_path = args[3]
    prompt = args[4]

    if method == 'llama':
        inference_llama(base_model_or_path, lora_weight_path, prompt)
    elif method == 'chatglm':
        inference_chatglm2(base_model_or_path, lora_weight_path, prompt)