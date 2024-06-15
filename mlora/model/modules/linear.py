from mlora.model.args import ModelData
from mlora.profiler import nvtx_range, set_backward_tracepoint

import torch
import bitsandbytes

from typing import Dict, Tuple, List

from .adapter import Adapter
from .lora import LoRAFunction


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

    def __lora_forward(self,
                       data: torch.Tensor,
                       input_args: ModelData,
                       result: torch.Tensor) -> torch.Tensor:
        # split the data and result
        dropouts: List[float] = []
        scalings: List[float] = []
        loras: Tuple[torch.Tensor] = ()

        for lora_config in input_args.data_config_:
            adapter_name = lora_config.adapter_name_

            if adapter_name == "" or adapter_name not in self.adapters_ or lora_config.adapter_type_ != "lora":
                loras += (None, None)
                dropouts.append(None)
                scalings.append(None)
                continue

            loras += (self.adapters_[adapter_name].lora_a_,
                      self.adapters_[adapter_name].lora_b_)
            dropouts.append(self.adapters_[adapter_name].dropout_)
            scalings.append(self.adapters_[adapter_name].scaling_)

        with nvtx_range("f_lora"):
            result = LoRAFunction.apply(
                result, data, input_args, dropouts, scalings, *loras)
        set_backward_tracepoint(result.grad_fn, "b_lora")

        return result

    def load_adapter(self, adapter: Adapter):
        assert adapter.adapter_name_ not in self.adapters_
        self.adapters_[adapter.adapter_name_] = adapter

    def offload_adapter(self, adapter_name: str):
        if adapter_name not in self.adapters_:
            return

        del self.adapters_[adapter_name]
