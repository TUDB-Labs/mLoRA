from mlora.modelargs import MultiLoraBatchData
from mlora.modelargs import LoraConfig

import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import bitsandbytes

from typing import Dict, Tuple


class Lora(nn.Module):
    def __init__(self,
                 base_layer: nn.Module,
                 shape: Tuple[int, int],
                 config: LoraConfig,
                 device: str):

        super().__init__()

        self.base_layer_ = base_layer
        self.device_ = torch.device(device)

        self.r_ = config.lora_r_
        self.alpha_ = config.lora_alpha_
        self.scaling_ = self.alpha_ / self.r_
        self.in_features_, self.out_features_ = shape

        if config.lora_dropout_ > 0.0:
            self.dropout_ = nn.Dropout(p=config.lora_dropout_)
        else:
            self.dropout_ = nn.Identity()

        self.lora_a_ = nn.Linear(
            self.in_features_, self.r_, bias=False, dtype=torch.float32, device=self.device_)
        self.lora_b_ = nn.Linear(
            self.r_, self.out_features_, bias=False, dtype=torch.float32, device=self.device_)

        self.use_dora_: bool = config.use_dora_
        self.magnitude_vector_: nn.Parameter = None

    def reset_parameters(self, lora_tensor=(None, None)):
        # if the lora_tensor is not (None, None), use it to init the lora weight
        assert isinstance(lora_tensor, Tuple)
        assert len(lora_tensor) == 2
        assert ((lora_tensor[0] is None) and (lora_tensor[1] is None)) or (isinstance(
            lora_tensor[0], torch.Tensor) and isinstance(lora_tensor[1], torch.Tensor))

        if lora_tensor == (None, None):
            nn.init.kaiming_uniform_(self.lora_a_.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b_.weight)
        else:
            with torch.no_grad():
                self.lora_a_.weight.copy_(lora_tensor[0])
                self.lora_b_.weight.copy_(lora_tensor[1])

        if self.use_dora_:
            self.dora_init()

    # Copied from HuggingFace PEFT
    def _get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1, dtype=torch.float32)
        return weight_norm

    def dora_init(self):
        lora_weight = self.lora_b_.weight @ self.lora_a_.weight
        weight_norm = self._get_weight_norm(
            self.base_layer_.weight, lora_weight, self.scaling_)
        self.magnitude_vector_ = nn.Parameter(
            weight_norm, requires_grad=True).to(self.device_)

    def _apply_dora(self, data: torch.Tensor):
        lora_weight = self.lora_b_.weight @ self.lora_a_.weight
        weight = self.base_layer_.weight
        weight_norm = self._get_weight_norm(
            weight, lora_weight, self.scaling_).detach()
        mag_norm_scale = (self.magnitude_vector_ / weight_norm).view(1, -1)
        # mikecovlee: setting fan_in_fan_out to False because we only use it for linear
        result_dora = (mag_norm_scale - 1) * (F.linear(data, weight).to(torch.float32)) + mag_norm_scale * \
            self.lora_b_(self.lora_a_(data.to(torch.float32))) * self.scaling_
        return result_dora

    def forward(self, data: torch.Tensor):
        result_dtype = data.dtype
        if self.use_dora_:
            result = self._apply_dora(self.dropout_(data))
        else:
            result = self.lora_b_(
                self.lora_a_(self.dropout_(data.to(torch.float32)))) * self.scaling_
        return result.to(result_dtype)


class Linear(nn.Module):
    def __init__(self, base_layer: nn.Module, device: str):
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            assert isinstance(base_layer, bitsandbytes.nn.Linear8bitLt) or isinstance(
                base_layer, bitsandbytes.nn.Linear4bit), f"error type - {type(base_layer)}."
        else:
            base_layer.requires_grad_(False)

        self.device_ = torch.device(device)
        self.base_layer_ = base_layer.to(self.device_)
        self.loras_: Dict[str, Lora] = {}

    def init_lora_weight(self,
                         lora_config: LoraConfig,
                         lora_tensor=(None, None),
                         adapter_name=None):
        if adapter_name is None:
            adapter_name = lora_config.adapter_name_

        if isinstance(self.base_layer_, bitsandbytes.nn.Linear4bit):
            out_dim, in_dim = self.base_layer_.out_features, self.base_layer_.in_features
        else:
            out_dim, in_dim = self.base_layer_.weight.shape

        if adapter_name not in self.loras_:
            self.loras_[adapter_name] = Lora(
                self.base_layer_, (in_dim, out_dim), lora_config, self.device_)

        self.loras_[adapter_name].reset_parameters(lora_tensor)

    def forward(self, data: torch.Tensor, input_args: MultiLoraBatchData) -> torch.Tensor:
        # data shape is: batch_size * max_seq_len * dim
        # result = data @ self.weight_.transpose(0, 1)
        result = self.base_layer_.forward(data)

        if len(self.loras_) == 0:
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
