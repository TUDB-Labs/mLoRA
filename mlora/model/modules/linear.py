from typing import Dict, List, Optional, Tuple
from mlora.utils import is_package_available

if is_package_available("bitsandbytes"):
    from bitsandbytes.nn import Linear8bitLt, Linear4bit
else:
    from mlora.utils import Linear8bitLt, Linear4bit

import torch

from mlora.model.args import ModelData
from mlora.profiler import nvtx_range, set_backward_tracepoint

from .adapter import Adapter
from .lora import LoRA, DoRA, LoRAFunction, get_range_tensor


class Linear(torch.nn.Module):
    def __init__(self, weight: torch.nn.Module):
        # the weight just wrapper the module from LlamaForCausalLM
        # the name for debug
        super().__init__()

        if not isinstance(weight, torch.nn.Linear):
            assert isinstance(weight, Linear8bitLt) or isinstance(
                weight, Linear4bit
            ), f"error type - {type(weight)}."
        else:
            weight.requires_grad_(False)

        self.device_ = weight.weight.device
        self.weight_ = weight
        self.adapters_: Dict[str, Adapter] = {}

    def forward(self, data: torch.Tensor, input_args: ModelData) -> torch.Tensor:
        # data shape is: batch_size * max_seq_len * dim
        # result = data @ self.weight_.transpose(0, 1)
        if len(self.adapters_) == 0:
            return self.weight_.forward(data)

        with nvtx_range("f_linear"):
            result = self.weight_.forward(data)
        set_backward_tracepoint(result.grad_fn, "b_linear")

        return self.__lora_forward(data, input_args, result)

    def _appy_dora(self,
                   residual: torch.Tensor,
                   lora_delta: torch.Tensor,
                   hidden_states: torch.Tensor,
                   input_args: ModelData):
        next_states = torch.zeros_like(residual)
        lora_range = get_range_tensor(
            next_states.device, batch_size=next_states.shape[0])
        for lora_config in input_args.lora_batch_data_config_:
            adapter_name = lora_config.adapter_name_
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            if adapter_name == "" or adapter_name not in self.loras_:
                continue

            if isinstance(self.loras_[adapter_name], DoRA):
                next_states.index_add_(0, lora_range[start_idx:end_idx], self.loras_[adapter_name].apply_dora(
                    residual[start_idx:end_idx], lora_delta[start_idx:end_idx], hidden_states[start_idx:end_idx]))
            else:
                next_states.index_add_(
                    0, lora_range[start_idx:end_idx], residual[start_idx:end_idx] + lora_delta[start_idx:end_idx])

        return next_states

    def __lora_forward(
        self, hidden_states: torch.Tensor, input_args: ModelData, residual: torch.Tensor
    ) -> torch.Tensor:
        # split the data and result
        dropouts: List[Optional[float]] = []
        scalings: List[Optional[float]] = []
        loras: Tuple[torch.Tensor | None, ...] = ()

        for lora_config in input_args.data_config_:
            adapter_name = lora_config.adapter_name_

            if adapter_name not in self.adapters_ or not isinstance(
                self.adapters_[adapter_name], LoRA
            ):
                loras += (None, None)
                dropouts.append(None)
                scalings.append(None)
                continue

            loras += (
                self.adapters_[adapter_name].lora_a_,
                self.adapters_[adapter_name].lora_b_,
            )
            dropouts.append(self.adapters_[adapter_name].dropout_)
            scalings.append(self.adapters_[adapter_name].scaling_)

        have_dora = any(isinstance(lora, DoRA)
                        for lora in self.loras_.values())

        if have_dora:
            lora_delta = torch.zeros_like(residual, dtype=torch.float32)
            with nvtx_range("f_lora"):
                lora_delta = LoRAFunction.apply(
                    lora_delta, hidden_states.to(torch.float32), input_args, dropouts, scalings, *loras)
            next_states = self._appy_dora(residual.to(
                torch.float32), lora_delta, hidden_states, input_args)
        else:
            with nvtx_range("f_lora"):
                next_states = LoRAFunction.apply(
                    residual.to(torch.float32), hidden_states.to(torch.float32), input_args, dropouts, scalings, *loras)

        set_backward_tracepoint(next_states.grad_fn, "b_lora")

        return next_states.to(hidden_states.dtype)

    def load_adapter(self, adapter: Adapter):
        assert adapter.adapter_name_ not in self.adapters_
        self.adapters_[adapter.adapter_name_] = adapter

    def offload_adapter(self, adapter_name: str):
        if adapter_name not in self.adapters_:
            return

        del self.adapters_[adapter_name]
