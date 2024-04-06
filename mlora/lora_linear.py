from mlora.modelargs import MultiLoraBatchData
from mlora.modelargs import LoraConfig
from mlora.utils import _is_package_available

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

if _is_package_available("bitsandbytes"):
    from bitsandbytes.nn import Linear8bitLt, Linear4bit
    import bitsandbytes as bnb
else:
    from mlora.utils import Linear8bitLt, Linear4bit

from typing import Dict, Tuple, List


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


g_cached_range_tensor: Dict[torch.device, torch.Tensor] = {}
# also max batch size
g_max_range = 128


def get_range_tensor(device: torch.device, batch_size: int = 1024):
    global g_cached_range_tensor
    global g_max_range
    if device not in g_cached_range_tensor or batch_size > g_max_range:
        g_max_range = g_max_range if g_max_range > batch_size else batch_size
        g_cached_range_tensor[device] = torch.arange(
            0, g_max_range, step=1, device=device)
    return g_cached_range_tensor[device]


class LoraFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                result: torch.Tensor,
                data: torch.Tensor,
                input_args: MultiLoraBatchData,
                dropouts: List[float],
                scalings: List[float],
                *args):
        save_inputs = (data,)

        lora_range = get_range_tensor(data.device, data.shape[0])
        for lora_a, lora_b, lora_config, dropout, scaling in zip(args[::2],
                                                                 args[1::2],
                                                                 input_args.lora_batch_data_config_,
                                                                 dropouts,
                                                                 scalings):
            assert not ((lora_a is None) ^ (lora_b is None))
            if lora_a is None and lora_b is None:
                save_inputs += (lora_a, lora_b, None)
                continue

            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            # must ensure the dropout is not zero
            # is dropout == 0, dropdata is a data's referece, so the data will be changed
            assert dropout > 0.0

            drop_data = F.dropout(
                data[start_idx:end_idx], p=dropout)
            drop_data.mul_(scaling)
            drop_data = drop_data @ lora_a.transpose(0, 1)

            lora_data = drop_data @ lora_b.transpose(0, 1)

            result.index_add_(
                dim=0, index=lora_range[start_idx:end_idx], source=lora_data)

            save_inputs += (lora_a, lora_b, drop_data)

        ctx.input_args = input_args
        ctx.dropouts = dropouts
        ctx.scalings = scalings
        ctx.save_for_backward(*save_inputs)

        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_result = None
        grad_data = None
        grad_input_args = None
        grad_dropouts = None
        grad_scalings = None
        grad_loras = ()

        data = ctx.saved_tensors[0]
        loras = ctx.saved_tensors[1:]

        if ctx.needs_input_grad[0]:
            grad_result = grad_output
        if ctx.needs_input_grad[1]:
            grad_data = torch.empty_like(data)

        lora_range = get_range_tensor(
            grad_output.device, batch_size=grad_output.shape[0])
        for lora_a, lora_b, drop_data, dropout, scaling, lora_config in zip(loras[::3],
                                                                            loras[1::3],
                                                                            loras[2::3],
                                                                            ctx.dropouts,
                                                                            ctx.scalings,
                                                                            ctx.input_args.lora_batch_data_config_):
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_
            assert not ((lora_a is None) ^ (lora_b is None))
            if lora_a is None and lora_b is None:
                grad_loras += (None, None)
                grad_data.index_fill_(
                    dim=0, index=lora_range[start_idx:end_idx], value=0)
                continue

            # lora_data shape is batch_size * seq_len * in_dim
            lora_data = data[start_idx:end_idx]
            # grad_y shape is batch_size * seq_len * out_dim
            grad_y = grad_output[start_idx:end_idx]

            # drop_data shape is batch_size * seq_len * r

            # bstage shape is batch_size * seq_len * r
            bstage = grad_y @ lora_b
            bstage *= (scaling / (1 - dropout))

            grad_a = torch.sum(bstage.transpose(1, 2) @ lora_data, dim=0)
            grad_b = torch.sum(grad_y.transpose(1, 2) @ drop_data, dim=0)
            grad_loras += (grad_a, grad_b)

            # grad_data shape is batch_size * seq_len * in_dim
            if grad_data is not None:
                grad_data[start_idx:end_idx] = bstage @ lora_a

        return grad_result, grad_data, grad_input_args, grad_dropouts, grad_scalings, *grad_loras


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

        assert config.lora_dropout_ > 0.0
        self.dropout_ = nn.Dropout(p=config.lora_dropout_)

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

    def apply_dora(
            self, residual: torch.Tensor, result_lora: torch.Tensor, hidden_states: torch.Tensor):
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
        return (mag_norm_scale - 1) * residual + mag_norm_scale * result_lora

    def forward(self, residual: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        result_lora = self.lora_b_(self.lora_a_(self.dropout_(
            hidden_states.to(torch.float32)))) * self.scaling_
        if self.use_dora_:
            return self.apply_dora(residual, result_lora, hidden_states)
        else:
            return residual + result_lora.to(residual.dtype)


class Linear(nn.Module):
    def __init__(self, base_layer: nn.Module, device: str):
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            assert isinstance(base_layer, Linear8bitLt) or isinstance(
                base_layer, Linear4bit), f"error type - {type(base_layer)}."
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

        if isinstance(self.base_layer_, Linear4bit):
            out_dim, in_dim = self.base_layer_.out_features, self.base_layer_.in_features
        else:
            out_dim, in_dim = self.base_layer_.weight.shape

        if adapter_name not in self.loras_:
            self.loras_[adapter_name] = Lora(
                self.base_layer_, (in_dim, out_dim), lora_config, self.device_)

        self.loras_[adapter_name].reset_parameters(lora_tensor)

    def _appy_dora(self,
                   residual: torch.Tensor,
                   lora_delta: torch.Tensor,
                   hidden_states: torch.Tensor,
                   input_args: MultiLoraBatchData):
        next_states = residual.clone()
        for lora_config in input_args.lora_batch_data_config_:
            adapter_name = lora_config.adapter_name_
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            if adapter_name == "" or adapter_name not in self.loras_:
                continue

            lora_range = get_range_tensor(
                next_states.device, batch_size=next_states.shape[0])[start_idx:end_idx]
            if self.loras_[adapter_name].use_dora_:
                next_states.index_add_(0, lora_range, self.loras_[adapter_name].apply_dora(
                    residual[start_idx:end_idx], lora_delta[start_idx:end_idx], hidden_states[start_idx:end_idx]))
            else:
                next_states.index_add_(
                    0, lora_range, lora_delta[start_idx:end_idx])

        return next_states

    def forward(self, hidden_states: torch.Tensor, input_args: MultiLoraBatchData) -> torch.Tensor:
        # hidden_states shape is: batch_size * max_seq_len * dim
        # result = hidden_states @ self.weight_.transpose(0, 1)
        residual = self.base_layer_.forward(hidden_states)

        if len(self.loras_) == 0:
            return residual

        # split the data and result
        dropouts: List[float] = []
        scalings: List[float] = []
        loras: Tuple[torch.Tensor] = ()
        for lora_config in input_args.lora_batch_data_config_:
            adapter_name = lora_config.adapter_name_

            if adapter_name == "" or adapter_name not in self.loras_:
                loras += (None, None)
                dropouts.append(None)
                scalings.append(None)
                continue

            loras += (self.loras_[adapter_name].lora_a_.weight,
                      self.loras_[adapter_name].lora_b_.weight)
            dropouts.append(self.loras_[adapter_name].dropout_.p)
            scalings.append(self.loras_[adapter_name].scaling_)

        have_dora = any(lora.use_dora_ for lora in self.loras_.values())

        if have_dora:
            lora_delta = torch.zeros_like(residual, dtype=torch.float32)
            lora_delta = LoraFunction.apply(
                lora_delta, hidden_states.to(torch.float32), input_args, dropouts, scalings, *loras)
            next_states = self._appy_dora(residual.to(
                torch.float32), lora_delta, hidden_states, input_args)
        else:
            next_states = LoraFunction.apply(
                residual.to(torch.float32), hidden_states.to(torch.float32), input_args, dropouts, scalings, *loras)

        return next_states.to(hidden_states.dtype)
