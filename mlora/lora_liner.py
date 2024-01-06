from mlora.modelargs import MultiLoraBatchData

import math
import torch
import torch.nn.functional as F
import bitsandbytes

from typing import Dict, Optional


class Lora(torch.nn.Module):
    def __init__(self, adapter_name: str):
        super().__init__()

        self.adapter_name_: str = adapter_name

        self.lora_a_: torch.Tensor = None
        self.lora_b_: torch.Tensor = None

        self.r_: int = 0
        self.alpha_: int = 0
        self.dropout_: float = 0.0
        self.scaling_: float = 0.0

    def set_parameter(self, r: int, alpha: int, dropout: float):
        self.r_ = r
        self.alpha_ = alpha
        self.dropout_ = dropout
        self.scaling_ = alpha / r

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data_ = F.dropout(data, self.dropout_)
        data_ @= self.lora_a_.transpose(0, 1)
        data_ @= self.lora_b_.transpose(0, 1)
        data_ *= self.scaling_
        return data_


class Linear(torch.nn.Module):
    def __init__(self, weight: torch.nn.Module, device: str = None):
        super().__init__()

        if device is None:
            self.device_ = weight.device
        else:
            self.device_ = device

        if not isinstance(weight, torch.nn.Linear):
            import bitsandbytes
            assert isinstance(weight,
                              bitsandbytes.nn.Linear8bitLt) or isinstance(weight,
                                                                          bitsandbytes.nn.Linear4bit), "error type."
        else:
            weight.requires_grad_(False)

        self.weight_ = weight.to(device)
        self.enable_lora_: bool = False
        self.loras_: Dict[str, Lora] = {}

    def init_lora_weight(self, adapter_name: str, r: int, alpha: int, dropout: float,
                         lora_a: Optional[torch.Tensor] = None,
                         lora_b: Optional[torch.Tensor] = None):
        if adapter_name not in self.loras_:
            self.loras_[adapter_name] = Lora(adapter_name)

        if isinstance(self.weight_, bitsandbytes.nn.Linear4bit):
            out_dim = self.weight_.out_features
            in_dim = self.weight_.in_features
        else:
            out_dim, in_dim = self.weight_.weight.shape

        self.loras_[adapter_name].set_parameter(r, alpha, dropout)

        if lora_a is not None:
            self.loras_[adapter_name].lora_a_ = lora_a.to(
                device=self.device_).to(torch.float32).requires_grad_(True)
        else:
            self.loras_[adapter_name].lora_a_ = torch.zeros(
                size=(r, in_dim), device=self.device_, requires_grad=True, dtype=torch.float32)
            torch.nn.init.kaiming_normal_(
                self.loras_[adapter_name].lora_a_, a=math.sqrt(5))

        if lora_b is not None:
            self.loras_[adapter_name].lora_b_ = lora_b.to(
                device=self.device_).to(torch.float32).requires_grad_(True)
        else:
            self.loras_[adapter_name].lora_b_ = torch.zeros(
                size=(out_dim, r), device=self.device_, requires_grad=True, dtype=torch.float32)

        self.enable_lora_ = True

    def forward(self, data: torch.Tensor, input_args: MultiLoraBatchData) -> torch.Tensor:
        # data shape is: batch_size * max_seq_len * dim
        # result = data @ self.weight_.transpose(0, 1)
        result = self.weight_.forward(data)

        if not self.enable_lora_:
            return result

        for lora_config in input_args.lora_batch_data_config_:
            adapter_name = lora_config.adapter_name_
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            if adapter_name == "" or adapter_name not in self.loras_:
                continue

            result[start_idx: end_idx] += self.loras_[
                adapter_name].forward(data[start_idx:end_idx])

        return result
