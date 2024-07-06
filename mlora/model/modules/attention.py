import math
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from mlora.model.args import LinearInfo, LLMModelArgs, ModelData
from mlora.model.modules import AdapterModel
from mlora.profiler import nvtx_range, set_backward_tracepoint

from .linear import Linear


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # see the above ref
    left_part = x[..., : x.shape[-1] // 2]
    right_part = x[..., x.shape[-1] // 2 :]
    return torch.cat((-right_part, left_part), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    # data shape is: batch_size * n_head * seq_len * n_dim
    xq_embed = (xq * cos) + (rotate_half(xq) * sin)
    xk_embed = (xk * cos) + (rotate_half(xk) * sin)
    return (xq_embed, xk_embed)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, n_kv_heads, seq_len, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
    x = x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)
    return x


def precompute_rope_angle(
    dim: int, seq_len: int, theta: float, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    # this implement is different with facebooksearch/llama
    #   ref: https://github.com/huggingface/transformers/issues/25199
    angles = 1.0 / (theta ** (torch.arange(0, dim, 2).float().to(device) / dim))
    seq = torch.arange(seq_len, device=device, dtype=angles.dtype)
    emb = torch.outer(seq, angles)
    emb = torch.cat((emb, emb), dim=-1)

    emb.requires_grad_(False)
    # cos(angle), sin(angle)
    return (emb.cos(), emb.sin())


@torch.jit.script
def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attention_score = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(
        query.size(-1)
    )
    if attention_mask is not None:
        attention_score = attention_score + attention_mask
    attention_score = F.softmax(attention_score, dim=-1, dtype=torch.float32).to(
        value.dtype
    )
    attention_score = torch.matmul(attention_score, value)
    attention_score = attention_score.transpose(1, 2).contiguous()
    return attention_score


class Attention(torch.nn.Module):
    wq_: Linear
    wk_: Linear
    wv_: Linear
    wo_: Linear

    def __init__(self, layer_id: int, args: LLMModelArgs):
        super().__init__()

        # use layer id to local the adapter
        self.layer_id_: int = layer_id

        self.n_heads_ = args.n_heads_
        self.n_kv_heads_ = args.n_kv_heads_
        self.head_dim_ = args.dim_ // args.n_heads_
        self.n_rep_ = self.n_heads_ // self.n_kv_heads_

        # rope angle cos and sin
        self.cos_, self.sin_ = precompute_rope_angle(
            args.dim_ // args.n_heads_,
            args.max_seq_len_,
            args.rope_theta_,
            args.device_,
        )

    def forward(self, data: torch.Tensor, mask: torch.Tensor, input_args: ModelData):
        batch_size, max_seq_len, _ = data.shape

        xq = self.wq_.forward(data, input_args)
        xk = self.wk_.forward(data, input_args)
        xv = self.wv_.forward(data, input_args)

        # conver shape to multi head
        # the shape is batch_size * number_of_head * seq_len * dim_of_head
        xq = xq.view(batch_size, max_seq_len, self.n_heads_, self.head_dim_).transpose(
            1, 2
        )
        xk = xk.view(
            batch_size, max_seq_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)
        xv = xv.view(
            batch_size, max_seq_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)

        # apply rotary embedding
        assert xq.dtype == xk.dtype
        cos = self.cos_[:max_seq_len].to(xq.dtype)
        sin = self.sin_[:max_seq_len].to(xq.dtype)

        with nvtx_range("f_rotray_emb"):
            xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        set_backward_tracepoint(xq.grad_fn, "b_q_rope")
        set_backward_tracepoint(xk.grad_fn, "b_k_rope")

        # for llama2 need to repeat the heads
        # before dim: batch_size, n_kv_head, seq_len, head_dim
        # after dim: batch_size, n_head, seq_len, head_dim
        xk = repeat_kv(xk, self.n_rep_)
        xv = repeat_kv(xv, self.n_rep_)

        set_backward_tracepoint(xk.grad_fn, "b_k_rep")
        set_backward_tracepoint(xv.grad_fn, "b_v_rep")

        # must align with xformers memory efficient attention
        with nvtx_range("f_attention"):
            attention_score = scaled_dot_product_attention(xq, xk, xv, mask)
        attention_score = attention_score.view(batch_size, max_seq_len, -1)
        set_backward_tracepoint(attention_score.grad_fn, "b_attention")

        # get output attention score
        return self.wo_.forward(attention_score, input_args)

    def from_pretrained(self, attn_layer: torch.nn.Module):
        self.wq_ = Linear(attn_layer.q_proj)
        self.wk_ = Linear(attn_layer.k_proj)
        self.wv_ = Linear(attn_layer.v_proj)
        self.wo_ = Linear(attn_layer.o_proj)

    @property
    def linear_dict(self) -> Dict[str, Linear]:
        return {
            f"layers.{self.layer_id_}.self_attn.q_proj": self.wq_,
            f"layers.{self.layer_id_}.self_attn.k_proj": self.wk_,
            f"layers.{self.layer_id_}.self_attn.v_proj": self.wv_,
            f"layers.{self.layer_id_}.self_attn.o_proj": self.wo_,
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
