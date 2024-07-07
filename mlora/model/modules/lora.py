import math
from typing import Any, Dict, List, Optional, Tuple, override

import torch
import torch.nn.functional as F

from mlora.backends import MPSBackend, get_backend
from mlora.model.args import ModelData

from .adapter import Adapter

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

            assert not ((lora_a.requires_grad) ^ (lora_b.requires_grad))
            if not lora_a.requires_grad and not lora_b.requires_grad:
                save_inputs += (None, None, None)
                continue

            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            # must ensure the dropout is not zero
            # is dropout == 0, dropdata is a data's referece
            # so the data will be changed
            assert dropout > 0.0

            drop_data = F.dropout(data[start_idx:end_idx], p=dropout)
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
    def init_grad_data(data: torch.Tensor) -> torch.Tensor:
        # mps do not support mps backend
        if isinstance(get_backend(), MPSBackend):
            return torch.zeros_like(data)
        else:
            return torch.empty_like(data)

    @staticmethod
    def in_place_fill_grad_data(grad_data: Optional[torch.Tensor], index: torch.Tensor):
        # mps use zero like, do not need to fill it again
        if grad_data is not None and not isinstance(get_backend(), MPSBackend):
            grad_data.index_fill_(0, index, 0)

    @staticmethod
    def in_place_add_grad_data(
        grad_data: torch.Tensor, index: torch.Tensor, grad_x: torch.Tensor
    ):
        # copy faster than add, but mps do not support copy
        if isinstance(get_backend(), MPSBackend):
            grad_data.index_add_(0, index, grad_x)
        else:
            grad_data.index_copy_(0, index, grad_x)

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
            grad_data = LoRAFunction.init_grad_data(data)

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
                # mps do not supprt empty like, so we need fill it
                LoRAFunction.in_place_fill_grad_data(
                    grad_data, lora_range[start_idx:end_idx]
                )
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
                grad_x = bstage @ lora_a
                LoRAFunction.in_place_add_grad_data(
                    grad_data, lora_range[start_idx:end_idx], grad_x
                )

        return (
            grad_result,
            grad_data,
            grad_input_args,
            grad_dropouts,
            grad_scalings,
            *grad_loras,
        )


class LoRA(Adapter):
    lora_a_: torch.Tensor
    lora_b_: torch.Tensor

    r_: int
    dropout_: float
    scaling_: float

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
        self.dropout_: float = dropout
        self.scaling_: float = alpha / r

    def init_weight(
        self, lora_a: torch.Tensor | None = None, lora_b: torch.Tensor | None = None
    ):
        # Gradient calculations are temporarily disabled for copy or init
        with torch.no_grad():
            if lora_a is None:
                torch.nn.init.kaiming_normal_(self.lora_a_, a=math.sqrt(5))
            else:
                self.lora_a_.copy_(lora_a)

            # lora_b is zero so do not need to init it
            if lora_b is not None:
                self.lora_b_.copy_(lora_b)

    @override
    def get_trainable_tensors(self) -> List[torch.Tensor]:
        return [self.lora_a_, self.lora_b_]

    @override
    def get_all_tensors(self) -> List[torch.Tensor]:
        return [self.lora_a_, self.lora_b_]
