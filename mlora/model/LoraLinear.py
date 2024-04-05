from mlora.model.modelargs import MultiLoraBatchData
from mlora.config import LoraConfig
from mlora.profiler.profiler import set_backward_tracepoint, nvtx_range

import math
import torch
import torch.nn.functional as F
import bitsandbytes

from typing import Dict, Optional, Tuple, List

g_cached_range_tensor: Dict[torch.device, torch.Tensor] = {}
# also max batch size
g_max_range = 1024


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
            assert dropout != 0

            drop_data = F.dropout(
                data[start_idx:end_idx], p=dropout, training=True, inplace=False)
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
                grad_x = bstage @ lora_a
                grad_data.index_copy_(
                    dim=0, index=lora_range[start_idx:end_idx], source=grad_x)

        return grad_result, grad_data, grad_input_args, grad_dropouts, grad_scalings, *grad_loras


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
        return LoraFunction.apply(data, self.lora_a_, self.lora_b_, self.dropout_, self.scaling_)


class Linear(torch.nn.Module):
    def __init__(self, weight: torch.nn.Module):
        # the weight just wrapper the module from LlamaForCausalLM
        # the name for debug
        super().__init__()

        if not isinstance(weight, torch.nn.Linear):
            assert isinstance(weight, bitsandbytes.nn.Linear8bitLt) or isinstance(
                weight, bitsandbytes.nn.Linear4bit), f"error type - {type(weight)}."
        else:
            weight.requires_grad_(False)

        self.device_ = weight.weight.device
        self.weight_ = weight
        self.enable_lora_: bool = False
        self.loras_: Dict[str, Lora] = {}

    def init_lora_weight(self,
                         lora_config: LoraConfig,
                         lora_tensor: Tuple[Optional[torch.Tensor],
                                            Optional[torch.Tensor]] = (None, None)):
        # if the lora_tensor is not (None, None), use it to init the lora weight
        assert isinstance(lora_tensor, Tuple)
        assert len(lora_tensor) == 2
        assert ((lora_tensor[0] is None) and (lora_tensor[1] is None)) or (isinstance(
            lora_tensor[0], torch.Tensor) and isinstance(lora_tensor[1], torch.Tensor))

        adapter_name = lora_config.adapter_name_
        r = lora_config.r_
        alpha = lora_config.lora_alpha_
        dropout = lora_config.lora_dropout_

        if adapter_name not in self.loras_:
            self.loras_[adapter_name] = Lora(adapter_name)
        self.loras_[adapter_name].set_parameter(r, alpha, dropout)

        if isinstance(self.weight_, bitsandbytes.nn.Linear4bit):
            out_dim, in_dim = self.weight_.out_features, self.weight_.in_features
        else:
            out_dim, in_dim = self.weight_.weight.shape

        def random_init_lora_a_tensor(lora: Lora):
            lora.__dict__["lora_a_"] = torch.zeros(
                size=(r, in_dim), device=self.device_, requires_grad=True, dtype=torch.float32)
            torch.nn.init.kaiming_normal_(lora.lora_a_, a=math.sqrt(5))

        def zero_init_lora_b_tensor(lora: Lora):
            lora.__dict__["lora_b_"] = torch.zeros(
                size=(out_dim, r), device=self.device_, requires_grad=True, dtype=torch.float32)

        def replace_init_lora_tensor(lora: Lora, lora_a: torch.Tensor, lora_b: torch.Tensor):
            lora.__dict__["lora_a_"] = lora_a.to(device=self.device_).to(
                torch.float32).detach().requires_grad_(True)
            lora.__dict__["lora_b_"] = lora_b.to(device=self.device_).to(
                torch.float32).detach().requires_grad_(True)

        # ensuer it's none, so we can use the __dict__ to init it
        assert self.loras_[adapter_name].lora_a_ is None
        assert self.loras_[adapter_name].lora_b_ is None

        if lora_tensor == (None, None):
            random_init_lora_a_tensor(self.loras_[adapter_name])
            zero_init_lora_b_tensor(self.loras_[adapter_name])
        else:
            replace_init_lora_tensor(self.loras_[adapter_name], *lora_tensor)

        self.enable_lora_ = True

    def forward(self, data: torch.Tensor, input_args: MultiLoraBatchData) -> torch.Tensor:
        # data shape is: batch_size * max_seq_len * dim
        # result = data @ self.weight_.transpose(0, 1)
        if not self.enable_lora_:
            return self.weight_.forward(data)

        with nvtx_range("f_linear"):
            result = self.weight_.forward(data)
        set_backward_tracepoint(result.grad_fn, "b_linear")

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

            loras += (self.loras_[adapter_name].lora_a_,
                      self.loras_[adapter_name].lora_b_)
            dropouts.append(self.loras_[adapter_name].dropout_)
            scalings.append(self.loras_[adapter_name].scaling_)

        with nvtx_range("f_lora"):
            result = LoraFunction.apply(
                result, data, input_args, dropouts, scalings, *loras)

        set_backward_tracepoint(result.grad_fn, "b_lora")

        return result
