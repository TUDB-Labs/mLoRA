from collections import OrderedDict
from typing import Dict

import torch
import torch.nn.functional as F

from mlora.model.args import LinearInfo, LLMModelArgs, ModelData
from mlora.model.modules import AdapterModel
from mlora.profiler import nvtx_range, set_backward_tracepoint
from mlora.config.config_chatglm import ChatGLMConfig

from .linear import Linear
from .attention import _config_to_kwargs


class MLP(torch.nn.Module):
    gate_: Linear  # also gate FNN * dim
    down_: Linear  # also down dim * FNN
    up_: Linear  # also up   FNN * dim

    def __init__(self, layer_id: int, config: LLMModelArgs):
        super().__init__()

        # use layer id to local the adapter
        self.layer_id_ = layer_id

    def forward(self, data: torch.Tensor, input_args: ModelData) -> torch.Tensor:
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
            f"layers.{self.layer_id_}.mlp.up_proj": self.up_,
        }

    def load_adapter(self, adapter_model: AdapterModel):
        for name, module in self.linear_dict().items():
            if name not in adapter_model:
                continue
            module.load_adapter(adapter_model[name])

    def offload_adapter(self, adapter_name: str):
        for _, module in self.linear_dict().items():
            module.offload_adapter(adapter_name)

    def linears_info(self) -> OrderedDict[str, LinearInfo]:
        ret_val = OrderedDict()

        for name, module in self.linear_dict().items():
            assert isinstance(module, Linear)
            ret_val[name] = LinearInfo(
                name_=name,
                in_dim_=module.weight_.in_features,
                out_dim_=module.weight_.out_features,
                base_weight_=module.weight_,
            )

        return ret_val


class ChatglmMLP(torch.nn.Module):
    dense_h_to_4h: Linear
    dense_4h_to_h: Linear
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, layer_id, config: ChatGLMConfig, device=None):
        super(ChatglmMLP, self).__init__()

        self.layer_id = layer_id
        self.add_bias = config.add_bias_linear

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu


    def forward(self, hidden_states, input_args: ModelData):
        # [s, b, 4hp]
        intermediate_parallel = self.dense_h_to_4h(hidden_states, input_args)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [s, b, h]
        output = self.dense_4h_to_h(intermediate_parallel, input_args)
        return output
    
    def from_pretrained(self, mlp_layer: torch.nn.Module) -> None:
        self.dense_h_to_4h = Linear(mlp_layer.dense_h_to_4h)
        self.dense_4h_to_h = Linear(mlp_layer.dense_4h_to_h)

    @property
    def linear_dict(self) -> Dict[str, Linear]:
        return {
            f"layers.{self.layer_id}.mlp.dense_h_to_4h": self.dense_h_to_4h,
            f"layers.{self.layer_id}.mlp.dense_4h_to_h": self.dense_4h_to_h,
        }

    def load_adapter(self, adapter_model: AdapterModel):
        for name, module in self.linear_dict.items():
            if name not in adapter_model:
                continue
            module.load_adapter(adapter_model[name])

    def offload_adapter(self, adapter_name: str):
        for _, module in self.linear_dict.items():
            module.offload_adapter(adapter_name)

    def linears_info(self) -> OrderedDict[str, LinearInfo]:
        ret_val = OrderedDict()

        for name, module in self.linear_dict.items():
            assert isinstance(module, Linear)
            ret_val[name] = LinearInfo(
                name_=name,
                in_dim_=module.weight_.in_features,
                out_dim_=module.weight_.out_features,
                base_weight_=module.weight_,
            )

        return ret_val
