from collections import OrderedDict

import torch

from mlora.model.args import LinearInfo, LLMModelArgs, ModelData
from mlora.model.modules import AdapterModel
from mlora.profiler import nvtx_range, set_backward_tracepoint

from .attention import Attention
from .mlp import MLP
from .rms_norm import RMSNorm


class Decoder(torch.nn.Module):
    attn_norm_: RMSNorm
    mlp_norm_: RMSNorm

    def __init__(self, layer_id: int, args: LLMModelArgs):
        super().__init__()

        self.layer_id_ = layer_id

        self.attn_: Attention = Attention(layer_id, args)
        self.mlp_: MLP = MLP(layer_id)

    def forward(
        self, hidden_states: torch.Tensor, mask: torch.Tensor, input_args: ModelData
    ):
        # Attention
        with nvtx_range("f_attention_norm"):
            attn_norm_output = self.attn_norm_.forward(hidden_states)
        set_backward_tracepoint(attn_norm_output.grad_fn, "b_attention_norm")

        attn_output = self.attn_.forward(attn_norm_output, mask, input_args)

        with nvtx_range("f_attn_add"):
            hidden_states = attn_output + hidden_states
        set_backward_tracepoint(attn_output.grad_fn, "b_attn_add")

        # MLP
        with nvtx_range("f_mlp_norm"):
            mlp_norm_output = self.mlp_norm_.forward(hidden_states)
        set_backward_tracepoint(mlp_norm_output.grad_fn, "b_mlp_norm")

        mlp_output = self.mlp_.forward(mlp_norm_output, input_args)
        with nvtx_range("f_mlp_add"):
            hidden_states = mlp_output + hidden_states
        set_backward_tracepoint(mlp_output.grad_fn, "b_mlp_add")

        return hidden_states

    def from_pretrained(
        self, transformer_layer: torch.nn.Module, norm_eps: float
    ) -> None:
        self.mlp_norm_ = RMSNorm(
            transformer_layer.post_attention_layernorm.weight, norm_eps
        )
        self.attn_norm_ = RMSNorm(transformer_layer.input_layernorm.weight, norm_eps)

        self.attn_.from_pretrained(transformer_layer.self_attn)
        self.mlp_.from_pretrained(transformer_layer.mlp)

    def load_adapter(self, adapter_model: AdapterModel):
        self.attn_.load_adapter(adapter_model)
        self.mlp_.load_adapter(adapter_model)

    def offload_adapter(self, adapter_name: str):
        self.attn_.offload_adapter(adapter_name)
        self.mlp_.offload_adapter(adapter_name)

    def linears_info(self) -> OrderedDict[str, LinearInfo]:
        ret_val: OrderedDict[str, LinearInfo] = OrderedDict()
        ret_val.update(self.attn_.linears_info())
        ret_val.update(self.mlp_.linears_info())
        return ret_val
