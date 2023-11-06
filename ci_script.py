from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel
import torch
from peft import PeftModel
import sys


def inference_llama(base_model_name_or_path, lora_weights_path, prompt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    inference(tokenizer, model, device, prompt)


def inference_chatglm(base_model_name_or_path, lora_weights_path, prompt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(base_model_name_or_path, trust_remote_code=True).to(device)
    model = PeftModel.from_pretrained(model, lora_weights_path)
    inference(tokenizer, model, device, prompt)


def inference(tokenizer, model, device, prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.5
        )
        output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]

        print(f"Prompt:\n{prompt}\n")
        print(f"Generated:\n{output}")


if __name__ == '__main__':
    args = sys.argv
    model_type = args[1]
    base_model_or_path = args[2]
    lora_weight_path = args[3]
    prompt = args[4]

    if model_type == 'llama':
        inference_llama(base_model_or_path, lora_weight_path, prompt)
    elif model_type == 'chatglm':
        inference_chatglm(base_model_or_path, lora_weight_path, prompt)
    else:
        print(f"no such model type: {model_type}.")
