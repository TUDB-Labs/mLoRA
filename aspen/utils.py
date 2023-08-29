import torch
from transformers import LlamaForCausalLM


# convert huggingface model to pytorch model
def convert_hf_to_pth(source: str, dest: str):
    src_model = LlamaForCausalLM.from_pretrained(source)
    # src_model.eval()
    torch.save(src_model.state_dict(), dest)
