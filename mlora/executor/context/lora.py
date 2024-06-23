from mlora.config import LoRAConfig
from mlora.config.adapter import AdapterConfig
from mlora.model.modules import LoRA
from mlora.model.args import LinearInfo

import os
import torch
import logging
from typing import Dict
from collections import OrderedDict

from .context import TaskContext
from .train import TrainTaskContext
from .inference import InferenceTaskContext


def _load_lora_weight(obj: TaskContext,
                      config: LoRAConfig,
                      linears_info: OrderedDict[str, LinearInfo]):
    # init the weight
    for linear_name, linear_info in linears_info.items():
        target_name = linear_name.split('.')[3]
        if target_name not in config.target_:
            continue
        if config.target_[target_name] is not True:
            continue

        obj.adapter_model_[linear_name] = LoRA(config.name_,
                                               linear_info.in_dim_, linear_info.out_dim_,
                                               config.r_, config.alpha_, config.dropout_)
    weight_dict = None
    temp_path=obj.path_
    if os.path.isdir(obj.path_):	    if os.path.isdir(os.path.join(obj.path_, "adapters")):
        temp_path=os.path.join(obj.path_, "adapters")
        folders = [folder for folder in os.listdir(temp_path)]
        temp_path=os.path.join(temp_path,folders[-1])

        last_cahr = temp_path[-1]
        last_digit = int(last_cahr)
        obj.now_step_= last_digit+1
        obj.last_epoch = last_digit
        #set now_step to the recently state before fault


    if os.path.isdir(temp_path):
        logging.info(
            f"Adapter {obj.name_}:{temp_path} weight exist, load from file.")
        weight_dict = torch.load(f"{temp_path}{os.sep}adapter_model.bin")
        prefix_name = "base_model.model.model."
    else:
        logging.info(
            f"Adapter {obj.name_}:{obj.path_} weight not exist, use the default weight.")

    for name, module in obj.adapter_model_.items():
        lora_a = None if weight_dict is None else weight_dict[prefix_name +
                                                              name + ".lora_A.weight"]
        lora_b = None if weight_dict is None else weight_dict[prefix_name +
                                                              name + ".lora_B.weight"]
        module.init_weight(lora_a, lora_b)


class InferenceLoRAContext(InferenceTaskContext):
    def __init__(self, config: AdapterConfig, linears_info: OrderedDict[str, LinearInfo]) -> None:
        super().__init__(config, linears_info)

    def load_weight(self, config: LoRAConfig, linears_info: OrderedDict[str, LinearInfo]):
        _load_lora_weight(self, config, linears_info)


class TrainLoRAContext(TrainTaskContext):
    def __init__(self, config: LoRAConfig, linears_info: OrderedDict[str, LinearInfo]) -> None:
        super().__init__(config, linears_info)

        self.loss_fn_ = torch.nn.CrossEntropyLoss()

    def load_weight(self, config: LoRAConfig, linears_info: OrderedDict[str, LinearInfo]):
        _load_lora_weight(self, config, linears_info)

    def weight_dict(self) -> Dict[str, torch.Tensor]:
        # base_model.model.model.layers.{0}.self_attn.{q_proj}.{lora_A}.weight
        # base_model.model.model.layers.{0}.mlp.{gate_proj}.{lora_A}.weight
        ret_val = {}
        for name, adapter in self.adapter_model_.items():
            prefix_name = "base_model.model.model." + name
            ret_val[prefix_name + ".lora_A.weight"] = adapter.lora_a_
            ret_val[prefix_name + ".lora_B.weight"] = adapter.lora_b_

        return ret_val
