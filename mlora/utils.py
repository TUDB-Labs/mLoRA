from mlora.model.model import LLMModel
from mlora.tokenizer.tokenizer import Tokenizer
from mlora.model.model_llama import LlamaModel
from mlora.model.model_chatglm import ChatGLMModel
from mlora.config import LoraConfig

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


def init_lora_model(llm_model: LLMModel, lora_configs: List[LoraConfig]):
    for lora_config in lora_configs:
        lora_weight = None
        adapter_file_path = lora_config.adapter_name_ + "/adapter_model.bin"

        if os.path.isfile(adapter_file_path):
            logging.info(f"load {adapter_file_path}")
            # Load adapter configuration for consistency check
            with open(lora_config.adapter_name_ + "/adapter_config.json", 'r', encoding='utf8') as fp:
                adapter_config = json.load(fp)
            base_model_name_or_path = adapter_config.get(
                "base_model_name_or_path", "")
            if base_model_name_or_path != "" and base_model_name_or_path != llm_model.name_or_path_:
                raise ValueError("loading adapter with unmatched base model." +
                                 f" current is {llm_model.name_or_path_}, provided {base_model_name_or_path}")
            # Load adapter weight
            lora_weight = torch.load(adapter_file_path)

        logging.info(
            f'init the lora adapter {lora_config.adapter_name_} weight.')

        llm_model.init_lora_weight(lora_config, lora_weight)
