import math
from typing import Any, Dict, List, Tuple, override
from mlora.utils import is_package_available

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlora.model.args import ModelData
if is_package_available("bitsandbytes"):
    import bitsandbytes as bnb

from .adapter import Adapter


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
g_max_range = 1024


def get_range_tensor(device: torch.device, batch_size: int = 1024):
    global g_cached_range_tensor
    global g_max_range
    if device not in g_cached_range_tensor or batch_size > g_max_range:
        g_max_range = g_max_range if g_max_range > batch_size else batch_size
        g_cached_range_tensor[device] = torch.arange(
            0, g_max_range, step=1, device=device
        )
    return g_cached_range_tensor[device]


class LoRAFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        result: torch.Tensor,
        data: torch.Tensor,
        input_args: ModelData,
        dropouts: List[float],
        scalings: List[float],
        *args,
    ):
        # the lora module is f32 precision
        data = data.to(torch.float32)

        save_inputs: Tuple[torch.Tensor | None, ...] = (data,)

        lora_range = get_range_tensor(data.device, data.shape[0])
        for lora_a, lora_b, lora_config, dropout, scaling in zip(
            args[::2], args[1::2], input_args.data_config_, dropouts, scalings
        ):
            assert not ((lora_a is None) ^ (lora_b is None))
            if lora_a is None and lora_b is None:
                save_inputs += (None, None, None)
                continue

            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            # must ensure the dropout is not zero
            # is dropout == 0
            #   dropdata is a data's referece, so the data will be changed
            assert dropout > 0.0

            drop_data = F.dropout(
                data[start_idx:end_idx], p=dropout)
            drop_data.mul_(scaling)

            drop_data = drop_data @ lora_a.transpose(0, 1)

            lora_data = drop_data @ lora_b.transpose(0, 1)

            lora_data = lora_data.to(result.dtype)

            result.index_add_(0, lora_range[start_idx:end_idx], lora_data)

            save_inputs += (lora_a, lora_b, drop_data)

        ctx.input_args = input_args
        ctx.dropouts = dropouts
        ctx.scalings = scalings
        ctx.save_for_backward(*save_inputs)

        return result

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output: torch.Tensor = grad_outputs[0]
        grad_result = None
        grad_data: torch.Tensor | None = None
        grad_input_args = None
        grad_dropouts = None
        grad_scalings = None
        grad_loras: Tuple[torch.Tensor | None, ...] = ()

        data, *loras = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            grad_result = grad_output
        if ctx.needs_input_grad[1]:
            grad_data = torch.zeros_like(data)

        # the lora module is fp32 precision
        grad_output = grad_output.to(torch.float32)
        lora_range = get_range_tensor(
            grad_output.device, batch_size=grad_output.shape[0]
        )
        for lora_a, lora_b, drop_data, dropout, scaling, lora_config in zip(
            loras[::3],
            loras[1::3],
            loras[2::3],
            ctx.dropouts,
            ctx.scalings,
            ctx.input_args.data_config_,
        ):
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_
            assert not ((lora_a is None) ^ (lora_b is None))
            if lora_a is None and lora_b is None:
                grad_loras += (None, None)
                continue

            # lora_data shape is batch_size * seq_len * in_dim
            lora_data = data[start_idx:end_idx]
            # grad_y shape is batch_size * seq_len * out_dim
            grad_y = grad_output[start_idx:end_idx]

            # bstage shape is batch_size * seq_len * r
            bstage = grad_y @ lora_b
            bstage *= scaling / (1 - dropout)

            grad_a = torch.sum(bstage.transpose(1, 2) @ lora_data, dim=0)
            grad_b = torch.sum(grad_y.transpose(1, 2) @ drop_data, dim=0)
            grad_loras += (grad_a, grad_b)

            # grad_data shape is batch_size * seq_len * in_dim
            if grad_data is not None:
                grad_data.index_add_(
                    0, lora_range[start_idx:end_idx], bstage @ lora_a)

        return (
            grad_result,
            grad_data,
            grad_input_args,
            grad_dropouts,
            grad_scalings,
            *grad_loras,
        )


class LoRA(Adapter):
    def __init__(
        self,
        adapter_name: str,
        in_dim: int,
        out_dim: int,
        r: int,
        alpha: int,
        dropout: float,
    ):
        super().__init__("lora", adapter_name)

        self.lora_a_: torch.Tensor = torch.zeros(
            size=(r, in_dim), device="cpu", requires_grad=True, dtype=torch.float32
        )
        self.lora_b_: torch.Tensor = torch.zeros(
            size=(out_dim, r), device="cpu", requires_grad=True, dtype=torch.float32
        )

        self.r_: int = r
        self.alpha_: int = alpha
        self.dropout_: float = dropout
        self.scaling_: float = alpha / r

    def init_weight(
        self, lora_a: torch.Tensor | None = None, lora_b: torch.Tensor | None = None
    ):
        if lora_a is None:
            torch.nn.init.kaiming_normal_(self.lora_a_, a=math.sqrt(5))
        else:
            self.lora_a_ = (
                lora_a.to("cpu")
                .detach()
                .clone()
                .to(dtype=torch.float32)
                .requires_grad_(True)
            )

        if lora_b is not None:
            self.lora_b_ = (
                lora_b.to("cpu")
                .detach()
                .clone()
                .to(dtype=torch.float32)
                .requires_grad_(True)
            )

    @override
    def get_tensors(self) -> List[torch.Tensor]:
        return [self.lora_a_, self.lora_b_]


class DoRA(LoRA):
    def __init__(self,
                 adapter_name: str,
                 base_layer: nn.Module,
                 in_dim: int,
                 out_dim: int,
                 r: int,
                 alpha: int,
                 dropout: float,
                 device: str):

        super().__init__(adapter_name, in_dim, out_dim, r, alpha, dropout)

        self.base_layer_ = base_layer
        self.device_ = torch.device(device)

    def _get_weight_norm(self, weight) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        lora_weight = self.lora_b_.weight @ self.lora_a_.weight
        weight = weight + self.scaling_ * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1, dtype=torch.float32)
        return weight_norm

    def init_weight(self, lora_a: torch.Tensor | None = None, lora_b: torch.Tensor | None = None):
        super().init_weight(lora_a, lora_b)
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
        return mag_norm_scale * residual + mag_norm_scale * result_lora
