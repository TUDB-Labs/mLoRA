from mlora.config import LoRAConfig
from mlora.model.modules import LoRA
from mlora.model.args import LinearInfo

import os
import torch
import logging
from typing import List, Dict
from collections import OrderedDict

from .context import TaskContext


class LoRATaskContext(TaskContext):
    def __init__(self, context_name: str) -> None:
        super().__init__("lora", context_name)

    def init(self, config: LoRAConfig,
             linears_info: OrderedDict[str, LinearInfo]) -> None:
        # init the adapter's weight
        for linear_name, linear_info in linears_info.items():
            target_name = linear_name.split('.')[3]
            if target_name not in config.target_:
                continue
            if config.target_[target_name] is not True:
                continue

            self.adapter_model_[linear_name] = LoRA(config.name_,
                                                    linear_info.in_dim_, linear_info.out_dim_,
                                                    config.r_, config.alpha_, config.dropout_)

        self.__init_weight()

        self.device_ = "cpu"

        # if optimizer is none, will not train this adapter
        if config.optimizer_config_.optimizer_ == "none":
            logging.info(
                f"Adapter {self.name_} no optimizer, maybe just for inference, so disable the grad.")
            return self.__disable_grad()

        # init the optimizer
        self.loss_fn_ = torch.nn.CrossEntropyLoss()
        self.optimizer_ = None
        self.lr_scheduler_ = None

        self.create_optimizer(
            self.__parameters(), config.optimizer_config_, config.lr_scheduler_config_)

    def __init_weight(self):
        weight_dict = None

        if os.path.isdir(self.name_):
            logging.info(f"Adapter {self.name_} weight exist, load from file.")
            weight_dict = torch.load(f"{self.name_}{os.sep}adapter_model.bin")
            prefix_name = "base_model.model.model."

        for name, module in self.adapter_model_.items():
            lora_a = None if weight_dict is None else weight_dict[prefix_name +
                                                                  name + ".lora_A.weight"]
            lora_b = None if weight_dict is None else weight_dict[prefix_name +
                                                                  name + ".lora_B.weight"]
            module.init_weight(lora_a, lora_b)

    def __parameters(self) -> List[torch.Tensor]:
        ret_val: List[torch.Tensor] = []
        for lora in self.adapter_model_.values():
            ret_val.append(lora.lora_a_)
            ret_val.append(lora.lora_b_)
        return ret_val

    def __disable_grad(self):
        for _, module in self.adapter_model_.items():
            module.disable_grad()

    def switch_device(self, device: str) -> None:
        if self.device_ == device:
            return

        for _, adapter in self.adapter_model_.items():
            self.switch_tensor(adapter.lora_a_, device)
            self.switch_tensor(adapter.lora_b_, device)

        self.switch_optimizer(device)

        self.device_ = device

    def step(self) -> None:
        self.optimizer_.step()
        if self.lr_scheduler_ is not None:
            self.lr_scheduler_.step()
        self.optimizer_.zero_grad()

    def weight_dict(self) -> Dict[str, torch.Tensor]:
        # base_model.model.model.layers.{0}.self_attn.{q_proj}.{lora_A}.weight
        # base_model.model.model.layers.{0}.mlp.{gate_proj}.{lora_A}.weight
        ret_val = {}
        for name, adapter in self.adapter_model_.items():
            prefix_name = "base_model.model.model." + name
            ret_val[prefix_name + ".lora_A.weight"] = adapter.lora_a_
            ret_val[prefix_name + ".lora_B.weight"] = adapter.lora_b_

        return ret_val
