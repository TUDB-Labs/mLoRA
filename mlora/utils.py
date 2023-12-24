from typing import Dict
from mlora.model_llama import LlamaModel
from mlora.model_mixlora import MixModel
from transformers import LlamaForCausalLM
import os
import json
import torch

# convert huggingface model to pytorch model


def convert_hf_to_pth(source: str, dest: str):
    src_model = LlamaForCausalLM.from_pretrained(source)
    # src_model.eval()
    torch.save(src_model.state_dict(), dest)

# save lora model


def save_lora_model(model: LlamaModel, config: Dict[str, str], dir_suffix=""):
    for lora_config in config["lora"]:
        lora_name = lora_config["name"]
        lora_output_dir = lora_config["output"]
        if dir_suffix != "":
            lora_output_dir += os.sep + \
                lora_config["output"] + "_" + dir_suffix

        if not os.path.exists(lora_output_dir):
            os.makedirs(lora_output_dir)

        lora_weight_dict, target_modules = model.get_lora_weight_dict(
            lora_name)

        torch.save(lora_weight_dict, lora_output_dir +
                   os.sep + "adapter_model.bin")

        adapter_config = {}
        adapter_config["lora_alpha"] = lora_config["alpha"]
        adapter_config["lora_dropout"] = lora_config["dropout"]
        adapter_config["r"] = lora_config["r"]
        adapter_config["peft_type"] = "LORA"
        adapter_config["task_type"] = "CAUSAL_LM"
        adapter_config["bias"] = "none"
        adapter_config["target_modules"] = target_modules

        with open(lora_output_dir + os.sep + "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=4)


def save_mixlora_model(model: MixModel, config: Dict[str, str], dir_suffix=""):
    for moe_config in config["lora"]:
        moe_name = moe_config["name"]
        output_dir = moe_config["output"]
        if dir_suffix != "":
            output_dir += os.sep + moe_config["output"] + "_" + dir_suffix

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        weight_dict, target_modules = model.get_moe_weight_dict(moe_name)
        torch.save(weight_dict, output_dir + os.sep + "mixlora_model.bin")

        mixlora_config = {}
        mixlora_config["lora_alpha"] = moe_config["alpha"]
        mixlora_config["lora_dropout"] = moe_config["dropout"]
        mixlora_config["r"] = moe_config["r"]
        mixlora_config["peft_type"] = "LORA"
        mixlora_config["task_type"] = "CAUSAL_LM"
        mixlora_config["bias"] = "none"
        mixlora_config["target_modules"] = target_modules
        mixlora_config["experts"] = moe_config["experts"]
        mixlora_config["topk"] = moe_config["topk"]

        with open(output_dir + os.sep + "mixlora_config.json", "w") as f:
            json.dump(mixlora_config, f, indent=4)
