from mlora.modelargs import MultiLoraBatchData
from mlora.modelargs import LoraConfig

import math
import torch
import torch.nn as nn
import bitsandbytes as bnb

from typing import Dict, Tuple


def is_quantized(weight: torch.nn.Parameter):
    cls_name = weight.__class__.__name__
    return cls_name in ("Params4bit", "Int8Params")


def dequantize_bnb_weight(weight: torch.nn.Parameter, state=None):
    cls_name = weight.__class__.__name__
    if cls_name not in ("Params4bit", "Int8Params"):
        return weight

    if cls_name == "Params4bit":
        return bnb.functional.dequantize_4bit(weight.data, weight.quant_state)

    if state.SCB is None:
        state.SCB = weight.SCB

    im = torch.eye(weight.data.shape[-1]).contiguous().half().to(weight.device)
    im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
    im, Sim = bnb.functional.transform(im, "col32")
    if state.CxB is None:
        state.CxB, state.SB = bnb.functional.transform(
            weight.data, to_order=state.formatB)
    out32, Sout32 = bnb.functional.igemmlt(im, state.CxB, Sim, state.SB)
    return bnb.functional.mm_dequant(out32, Sout32, SCim, state.SCB, bias=None).t()


class Lora(nn.Module):
    def __init__(self,
                 base_layer: nn.Module,
                 shape: Tuple[int, int],
                 config: LoraConfig,
                 device: str):

        super().__init__()

        self.base_layer_ = base_layer
        self.device_ = torch.device(device)

        self.initializer_ = config.lora_init_
        self.r_ = config.lora_r_
        self.alpha_ = config.lora_alpha_

        if config.use_rslora_:
            self.scaling_ = self.alpha_ / math.sqrt(self.r_)
        else:
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

    def _get_weight_norm(self, weight) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        lora_weight = self.lora_b_.weight @ self.lora_a_.weight
        weight = weight + self.scaling_ * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1, dtype=torch.float32)
        return weight_norm

    def reset_parameters(self, lora_tensor=(None, None)) -> None:
        # if the lora_tensor is not (None, None), use it to init the lora weight
        assert isinstance(lora_tensor, Tuple)
        assert len(lora_tensor) == 2
        assert ((lora_tensor[0] is None) and (lora_tensor[1] is None)) or (isinstance(
            lora_tensor[0], torch.Tensor) and isinstance(lora_tensor[1], torch.Tensor))

        if lora_tensor == (None, None):
            if self.initializer_ == "original":
                nn.init.kaiming_uniform_(self.lora_a_.weight, a=math.sqrt(5))
            elif self.initializer_ == "gaussian":
                nn.init.normal_(self.lora_a_.weight, std=1 / self.r_)
            else:
                raise ValueError(f"Unknown initialization {self.initializer_}")
            nn.init.zeros_(self.lora_b_.weight)
        else:
            with torch.no_grad():
                self.lora_a_.weight.copy_(lora_tensor[0])
                self.lora_b_.weight.copy_(lora_tensor[1])

        if self.use_dora_:
            weight = self.base_layer_.weight
            quant_state = getattr(self.base_layer_, "state", None)
            weight = dequantize_bnb_weight(weight, state=quant_state)
            self.magnitude_vector_ = nn.Parameter(
                self._get_weight_norm(weight), requires_grad=True).to(self.device_)

    def forward(self, residual: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = residual.to(torch.float32)
        result_lora = self.lora_b_(self.lora_a_(self.dropout_(
            hidden_states.to(torch.float32)))) * self.scaling_
        if self.use_dora_:
            weight = self.base_layer_.weight
            if is_quantized(weight):
                # for 8bit and 4bit quantization
                quant_state = getattr(self.base_layer_, "state", None)
                weight = dequantize_bnb_weight(
                    weight, state=quant_state).to(torch.float32)
                residual = torch.nn.functional.linear(
                    hidden_states.to(torch.float32), weight)
            else:
                # for full precision or half precision
                weight = weight.to(torch.float32)
            weight_norm = self._get_weight_norm(weight).detach()
            mag_norm_scale = (self.magnitude_vector_ / weight_norm).view(1, -1)
            result = mag_norm_scale * residual + mag_norm_scale * result_lora
        else:
            result = residual + result_lora
        return result.to(hidden_states.dtype)


class Linear(nn.Module):
    def __init__(self, base_layer: nn.Module, device: str):
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            assert isinstance(base_layer, bnb.nn.Linear8bitLt) or isinstance(
                base_layer, bnb.nn.Linear4bit), f"error type - {type(base_layer)}."
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
            adapter_name = lora_config.adapter_name

        if isinstance(self.base_layer_, bnb.nn.Linear4bit):
            out_dim, in_dim = self.base_layer_.out_features, self.base_layer_.in_features
        else:
            out_dim, in_dim = self.base_layer_.weight.shape

        if adapter_name not in self.loras_:
            self.loras_[adapter_name] = Lora(
                self.base_layer_, (in_dim, out_dim), lora_config, self.device_)

        self.loras_[adapter_name].reset_parameters(lora_tensor)

    def forward(self, hidden_states: torch.Tensor, input_args: MultiLoraBatchData) -> torch.Tensor:
        # hidden_states shape is: batch_size * max_seq_len * dim
        # result = hidden_states @ self.weight_.transpose(0, 1)
        result = self.base_layer_.forward(hidden_states)

        if len(self.loras_) == 0:
            return result

        for lora_config in input_args.lora_batch_data_config_:
            adapter_name = lora_config.adapter_name_
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            if adapter_name == "" or adapter_name not in self.loras_:
                continue

            result[start_idx: end_idx] = self.loras_[adapter_name].forward(
                result[start_idx: end_idx], hidden_states[start_idx:end_idx])

        return result
