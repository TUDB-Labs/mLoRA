from mlora.trainer.context import TaskContext
from mlora.model.args import MLoRABatchData, LinearInfo
from mlora.profiler import nvtx_range, set_backward_tracepoint

import torch
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict

from .linear import Linear


class MLP(torch.nn.Module):
    def __init__(self, layer_id: int):
        super().__init__()

        # use layer id to local the adapter
        self.layer_id_ = layer_id

        self.gate_: Linear = None  # also gate FNN * dim
        self.down_: Linear = None  # also down dim * FNN
        self.up_: Linear = None  # also up   FNN * dim

    def forward(self, data: torch.Tensor, input_args: MLoRABatchData) -> torch.Tensor:
        # feed forward fully connected
        with nvtx_range("f_mlp"):
            w1 = self.gate_.forward(data, input_args)
            w3 = self.up_.forward(data, input_args)
            # same as: data = data + w2_forward(F.silu(w1) * w3, input_args)
            w1_silu = F.silu(w1)
            mlp_output = w1_silu * w3
            mlp_output = self.down_.forward(mlp_output, input_args)
        set_backward_tracepoint(mlp_output.grad_fn, "b_mlp")

        return mlp_output

    def from_pretrained(self, mlp_layer: torch.nn.Module) -> None:
        self.gate_ = Linear(mlp_layer.gate_proj)
        self.down_ = Linear(mlp_layer.down_proj)
        self.up_ = Linear(mlp_layer.up_proj)

    @property
    def linear_dict(self) -> Dict[str, Linear]:
        return {
            f"layers.{self.layer_id_}.mlp.gate_proj": self.gate_,
            f"layers.{self.layer_id_}.mlp.down_proj": self.down_,
            f"layers.{self.layer_id_}.mlp.up_proj": self.up_
        }

    def load_adapter(self, context: TaskContext):
        for name, module in self.linear_dict.items():
            if name not in context.adapter_:
                continue
            module.load_adapter(context.adapter_[name])

    def offload_adapter(self, adapter_name: str):
        for _, module in self.linear_dict.items():
            module.offload_adapter(adapter_name)

    def linears_info(self) -> OrderedDict[str, LinearInfo]:
        ret_val = OrderedDict()

        for name, module in self.linear_dict.items():
            assert isinstance(module, Linear)
            ret_val[name] = LinearInfo(name_=name,
                                       in_dim_=module.weight_.in_features,
                                       out_dim_=module.weight_.out_features)

        return ret_val
