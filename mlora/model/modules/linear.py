from typing import Callable, List, MutableMapping, Optional, Tuple

import torch
import torch.nn.functional as F

from mlora.model.args import ModelData
from mlora.profiler import nvtx_range, set_backward_tracepoint
from mlora.utils import is_package_available

if is_package_available("bitsandbytes"):
    from bitsandbytes.nn import Linear4bit, Linear8bitLt
else:
    from mlora.utils import Linear8bitLt, Linear4bit

from .adapter import Adapter
from .dora import DoRA
from .lora import LoRA, LoRAFunction, get_range_tensor
from .vera import VeRA


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
        self.adapters_: MutableMapping[str, Adapter] = {}

    def forward(self, data: torch.Tensor, input_args: ModelData) -> torch.Tensor:
        # data shape is: batch_size * max_seq_len * dim
        # result = data @ self.weight_.transpose(0, 1)
        if len(self.adapters_) == 0:
            return self.weight_.forward(data)

        with nvtx_range("f_linear"):
            result = self.weight_.forward(data)
        set_backward_tracepoint(result.grad_fn, "b_linear")

        adapter_func_list: List[Callable] = [
            self.__lora_forward,
            self.__vera_forward,
            self.__dora_forward,
        ]

        for func in adapter_func_list:
            result = func(data, input_args, result)

        return result

    def __lora_forward(
        self, data: torch.Tensor, input_args: ModelData, result: torch.Tensor
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

        with nvtx_range("f_lora"):
            result = LoRAFunction.apply(
                result, data, input_args, dropouts, scalings, *loras
            )
        set_backward_tracepoint(result.grad_fn, "b_lora")

        return result

    def __vera_forward(
        self, data: torch.Tensor, input_args: ModelData, result: torch.Tensor
    ) -> torch.Tensor:
        lora_range = get_range_tensor(data.device, data.shape[0])

        for lora_config in input_args.data_config_:
            adapter_name = lora_config.adapter_name_

            if adapter_name not in self.adapters_ or not isinstance(
                self.adapters_[adapter_name], VeRA
            ):
                continue

            adapter = self.adapters_[adapter_name]

            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            with nvtx_range("f_vera"):
                lora_data = F.dropout(
                    data[start_idx:end_idx],
                    p=adapter.dropout_,
                    training=True,
                    inplace=False,
                )
                lora_data = lora_data.mul(adapter.scaling_)
                lora_data = lora_data @ adapter.lora_a_.transpose(0, 1)
                lora_data = lora_data * adapter.d_vec_
                lora_data = lora_data @ adapter.lora_b_.transpose(0, 1)
                lora_data = lora_data * adapter.b_vec_
                lora_data = lora_data.to(result.dtype)

                result = result.index_add(
                    dim=0, index=lora_range[start_idx:end_idx], source=lora_data
                )

        set_backward_tracepoint(result.grad_fn, "b_vera")

        return result

    def __dora_forward(
        self, data: torch.Tensor, input_args: ModelData, result: torch.Tensor
    ) -> torch.Tensor:
        lora_range = get_range_tensor(data.device, data.shape[0])

        for lora_config in input_args.data_config_:
            adapter_name = lora_config.adapter_name_

            if adapter_name not in self.adapters_ or not isinstance(
                self.adapters_[adapter_name], DoRA
            ):
                continue

            adapter = self.adapters_[adapter_name]

            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            with nvtx_range("f_dora"):
                weight_norm = adapter.get_weight_norm()
                mag_norm_scale = (adapter.magnitude_ / weight_norm).view(1, -1)

                dora_data = F.dropout(
                    data[start_idx:end_idx],
                    p=adapter.dropout_,
                    training=True,
                    inplace=False,
                )
                lora_result = dora_data @ adapter.lora_a_.transpose(0, 1)
                lora_result = lora_result @ adapter.lora_b_.transpose(0, 1)
                lora_result = mag_norm_scale * lora_result * adapter.scaling_

                base_result = (
                    result[start_idx:end_idx] * (mag_norm_scale - 1) + lora_result
                )

                result = result.index_copy(
                    dim=0, index=lora_range[start_idx:end_idx], source=base_result
                )

        set_backward_tracepoint(result.grad_fn, "b_dora")

        return result

    def load_adapter(self, adapter: Adapter):
        assert adapter.adapter_name_ not in self.adapters_
        self.adapters_[adapter.adapter_name_] = adapter

    def offload_adapter(self, adapter_name: str):
        if adapter_name not in self.adapters_:
            return

        del self.adapters_[adapter_name]
