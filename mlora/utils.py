from mlora.model.model import LLMModel
from mlora.tokenizer.tokenizer import Tokenizer
from mlora.model.model_llama import LlamaModel
from mlora.model.model_chatglm import ChatGLMModel
from transformers import LlamaForCausalLM

import os
import json
import torch
import random
import logging

from typing import Tuple, Dict, List


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


def save_lora_model(model: LLMModel, config: Dict[str, str], dir_suffix=""):
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


def load_base_model(base_model: str,
                    model_type: str,
                    device: str,
                    load_4bit: bool = False,
                    load_8bit: bool = False,
                    partial_model_to_device: List[int] = None) -> Tuple[Tokenizer, LLMModel]:
    # base_model: the base model name from huggingface or the path of the base model
    assert not (load_4bit and load_8bit)

    model_type_dict: Dict[str, LLMModel] = {
        "llama": LlamaModel,
        "chatglm": ChatGLMModel
    }

    assert model_type in model_type_dict, f"unkown model type {model_type}"

    bits = None
    bits = 8 if load_8bit else bits
    bits = 4 if load_4bit else bits

    model = model_type_dict[model_type].from_pretrained(path=base_model,
                                                        device=device,
                                                        bits=bits,
                                                        partial_model_to_device=partial_model_to_device)

    tokenizer = Tokenizer(base_model)

    model.pad_token_id_ = tokenizer.pad_id_

    return tokenizer, model


def init_lora_model(config: Dict[str, any],
                    llm_model: LLMModel,
                    load_lora: bool = False):
    for lora_config in config["lora"]:
        lora_weight = None
        if load_lora:
            adapter_file_path = lora_config["output"] + "/adapter_model.bin"
            print(f"load {adapter_file_path}")
            lora_weight = torch.load(adapter_file_path)

        logging.info(f'init the lora adapter {lora_config["name"]} weight.')
        llm_model.init_lora_weight(lora_config["name"],
                                   lora_config["r"],
                                   lora_config["alpha"],
                                   lora_config["dropout"],
                                   lora_config["target_modules"],
                                   lora_weight)
