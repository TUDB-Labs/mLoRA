from mlora.config import LoRAConfig
from mlora.model.modules import LoRA
from mlora.model.args import LinearInfo

import torch
from typing import List, Dict
from collections import OrderedDict

from .context import TaskContext


class LoRATaskContext(TaskContext):
    def __init__(self, context_name: str) -> None:
        super().__init__("lora", context_name)

    def init_adapter(self, config: LoRAConfig,
                     linears_info: OrderedDict[str, LinearInfo]) -> None:
        for linear_name, linear_info in linears_info.items():
            target_name = linear_name.split('.')[3]
            if target_name not in config.target_:
                continue
            if config.target_[target_name] is not True:
                continue

            self.adapter_[linear_name] = LoRA(config.name_,
                                              linear_info.in_dim_, linear_info.out_dim_,
                                              config.r_, config.alpha_, config.dropout_)
            self.adapter_[linear_name].init_weight()

        self.optim_config_ = config.optimizer_config_
        self.lr_schduler_config_ = config.lr_scheduler_config_

        self.optimizer_ = None
        self.lr_scheduler_ = None

        self.device_ = "cpu"

        self.create_optimizer(
            self.__parameters, config.optimizer_config_, config.lr_scheduler_config_)

    @property
    def __parameters(self) -> List[torch.Tensor]:
        ret_val: List[torch.Tensor] = []
        for lora in self.adapter_.values():
            ret_val.append(lora.lora_a_)
            ret_val.append(lora.lora_b_)
        return ret_val

    def switch_device(self, device: str) -> None:
        if self.device_ == device:
            return

        for _, adapter in self.adapter_.items():
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
        for name, adapter in self.adapter_.items():
            prefix_name = "base_model.model.model." + name
            ret_val[prefix_name+".lora_A.weight"] = adapter.lora_a_
            ret_val[prefix_name+".lora_B.weight"] = adapter.lora_b_

        return ret_val
