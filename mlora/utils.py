from typing import Dict
from mlora.model_llama import LlamaModel
from transformers import LlamaForCausalLM

import os
import json
import torch
import random
import logging


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def setup_logging(log_level: str = "INFO", log_file: str = None):
    # set the logger
    log_handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(format="[%(asctime)s] m-LoRA: %(message)s",
                        level=log_level,
                        handlers=log_handlers,
                        force=True)


def setup_cuda_check():
    # check the enviroment
    if torch.cuda.is_available():
        logging.info('NVIDIA CUDA initialized successfully.')
        logging.info('Total %i GPU(s) detected.' % torch.cuda.device_count())
    else:
        logging.error(
            'm-LoRA requires NVIDIA CUDA computing capacity. Please check your PyTorch installation.')
        exit(1)


def convert_hf_to_pth(source: str, dest: str):
    # convert huggingface model to pytorch model
    src_model = LlamaForCausalLM.from_pretrained(source)
    torch.save(src_model.state_dict(), dest)


def save_lora_model(model: LlamaModel, config: Dict[str, str], dir_suffix=""):
    # save lora model
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
