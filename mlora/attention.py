from mlora.lora_linear import Linear
from mlora.modelargs import LLMModelArgs, MultiLoraBatchData
from mlora.backends import _backend
from mlora.utils import _is_package_available

from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import math

if _is_package_available("xformers"):
    import xformers.ops
    import xformers.ops.fmha.attn_bias
    _xformers_available = True
else:
    _xformers_available = False

if _is_package_available("flash_attn"):
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    _flash_attn_available = True
else:
    _flash_attn_available = False


def precompute_rope_angle(dim: int, seq_len: int,
                          theta: float = 10000.0,
                          device: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2,
                      dtype=torch.int64).to(device=device, dtype=torch.float32) / dim))
    t = torch.arange(seq_len, dtype=torch.int64).to(
        device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    emb.requires_grad_(False)

    # cos(angle), sin(angle)
    return (emb.cos(), emb.sin())


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_embed = (xq * cos) + (rotate_half(xq) * sin)
    k_embed = (xk * cos) + (rotate_half(xk) * sin)
    return q_embed, k_embed


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(
        seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


@torch.jit.script
def _scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                                  attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    attention_score = torch.matmul(
        query, key.transpose(2, 3)) / math.sqrt(query.size(-1))
    if attention_mask is not None:
        attention_score = attention_score + attention_mask
    attention_score = F.softmax(
        attention_score, dim=-1, dtype=torch.float32).to(query.dtype)
    attention_score = torch.matmul(attention_score, value)
    attention_score = attention_score.transpose(1, 2).contiguous()
    return attention_score


# Multi-headed attention from 'Attention Is All You Need' paper.
class LlamaAttention(torch.nn.Module):
    def __init__(self, wq: Linear, wk: Linear, wv: Linear, wo: Linear,
                 args: LLMModelArgs, layer_idx: int):
        super().__init__()
        # attention
        self.wq_: Linear = wq  # dim * dim
        self.wk_: Linear = wk  # dim * dim
        self.wv_: Linear = wv  # dim * dim
        self.wo_: Linear = wo  # dim * dim
        # cos and sin
        self.cos_, self.sin_ = precompute_rope_angle(
            args.dim_ // args.n_heads_, args.max_seq_len_, args.rope_theta_, args.device_)
        # other arg
        self.layer_id_ = layer_idx
        self.dim_ = args.dim_
        self.n_heads_ = args.n_heads_
        self.n_kv_heads_ = args.n_kv_heads_
        self.n_rep_ = self.n_heads_ // self.n_kv_heads_
        self.head_dim_ = args.dim_ // args.n_heads_
        self.dtype_ = args.dtype_
        self.is_causal_ = True

    def forward(self,
                hidden_states: torch.Tensor,
                input_args: MultiLoraBatchData,
                attention_mask: Optional[torch.Tensor] = None):
        batch_size, max_seq_len, _ = hidden_states.shape

        xq = self.wq_.forward(hidden_states, input_args)
        xk = self.wk_.forward(hidden_states, input_args)
        xv = self.wv_.forward(hidden_states, input_args)

        # conver shape to multi head
        xq = xq.view(batch_size, max_seq_len, self.n_heads_,
                     self.head_dim_).transpose(1, 2)
        xk = xk.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)
        xv = xv.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)

        # apply rotary embedding
        assert xq.dtype == xk.dtype
        cos = self.cos_[:max_seq_len].to(xq.dtype)
        sin = self.sin_[:max_seq_len].to(xq.dtype)
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        # for llama2 need to repeat the heads
        # before dim: batch_size, n_kv_head, seq_len, head_dim
        # after dim: batch_size, n_head, seq_len, head_dim
        xk = repeat_kv(xk, self.n_rep_)
        xv = repeat_kv(xv, self.n_rep_)

        attention_score = _scaled_dot_product_attention(
            xq, xk, xv, attention_mask)

        attention_score = attention_score.reshape(batch_size, max_seq_len, -1)

        # get output attention score
        return self.wo_.forward(attention_score, input_args)


class LlamaXformersAttention(LlamaAttention):
    def __init__(self, wq: Linear, wk: Linear, wv: Linear, wo: Linear,
                 args: LLMModelArgs, layer_idx: int):
        assert _xformers_available, "xFormers Attention is not available"
        super().__init__(wq, wk, wv, wo, args, layer_idx)

    def _xformers_attention(self, xq, xk, xv, attention_mask):
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        attention_score = xformers.ops.memory_efficient_attention(
            xq, xk, xv, attention_mask)
        return attention_score

    def forward(self,
                hidden_states: torch.Tensor,
                input_args: MultiLoraBatchData,
                attention_mask: Optional[torch.Tensor] = None):
        batch_size, max_seq_len, _ = hidden_states.shape

        xq = self.wq_.forward(hidden_states, input_args)
        xk = self.wk_.forward(hidden_states, input_args)
        xv = self.wv_.forward(hidden_states, input_args)

        # conver shape to multi head
        xq = xq.view(batch_size, max_seq_len, self.n_heads_,
                     self.head_dim_).transpose(1, 2)
        xk = xk.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)
        xv = xv.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)

        # apply rotary embedding
        assert xq.dtype == xk.dtype
        cos = self.cos_[:max_seq_len].to(xq.dtype)
        sin = self.sin_[:max_seq_len].to(xq.dtype)
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        # for llama2 need to repeat the heads
        # before dim: batch_size, n_kv_head, seq_len, head_dim
        # after dim: batch_size, n_head, seq_len, head_dim
        xk = repeat_kv(xk, self.n_rep_)
        xv = repeat_kv(xv, self.n_rep_)

        attention_score = self._xformers_attention(
            xq, xk, xv, attention_mask)

        attention_score = attention_score.reshape(batch_size, max_seq_len, -1)

        # get output attention score
        return self.wo_.forward(attention_score, input_args)


class LlamaFlashAttention(LlamaAttention):
    def __init__(self, wq: Linear, wk: Linear, wv: Linear, wo: Linear,
                 args: LLMModelArgs, layer_idx: int):
        assert _flash_attn_available, "Flash Attention is not available"
        super().__init__(wq, wk, wv, wo, args, layer_idx)

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=self.is_causal_,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=self.is_causal_
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(
            attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len,
                              num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len,
                                num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len,
                                    self.n_heads_, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def forward(self,
                hidden_states: torch.Tensor,
                input_args: MultiLoraBatchData,
                attention_mask: Optional[torch.Tensor] = None):
        batch_size, max_seq_len, _ = hidden_states.shape

        xq = self.wq_.forward(hidden_states, input_args)
        xk = self.wk_.forward(hidden_states, input_args)
        xv = self.wv_.forward(hidden_states, input_args)

        # conver shape to multi head
        xq = xq.view(batch_size, max_seq_len, self.n_heads_,
                     self.head_dim_).transpose(1, 2)
        xk = xk.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)
        xv = xv.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)

        # apply rotary embedding
        assert xq.dtype == xk.dtype
        cos = self.cos_[:max_seq_len].to(xq.dtype)
        sin = self.sin_[:max_seq_len].to(xq.dtype)
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        # for llama2 need to repeat the heads
        # before dim: batch_size, n_kv_head, seq_len, head_dim
        # after dim: batch_size, n_head, seq_len, head_dim
        xk = repeat_kv(xk, self.n_rep_)
        xv = repeat_kv(xv, self.n_rep_)

        input_dtype = xq.dtype
        if input_dtype == torch.float32:
            if _backend.is_bf16_supported():
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float16
            xq = xq.to(target_dtype)
            xk = xk.to(target_dtype)
            xv = xv.to(target_dtype)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            xq,
            xk,
            xv,
            attention_mask,
            max_seq_len,
        ).to(input_dtype)

        attn_output = attn_output.reshape(
            batch_size, max_seq_len, self.dim_).contiguous()
        attn_output = self.wo_.forward(attn_output, input_args)

        return attn_output


class MistralFlashAttention(LlamaAttention):
    def __init__(self, wq: Linear, wk: Linear, wv: Linear, wo: Linear,
                 args: LLMModelArgs, layer_idx: int):
        assert _flash_attn_available, "Flash Attention is not available"
        super().__init__(wq, wk, wv, wo, args, layer_idx)
        # Qwen2
        self.use_sliding_window_ = args.use_sliding_window_
        self.max_window_layers_ = args.max_window_layers_
        # Mistral and Qwen2
        self.sliding_window_ = args.sliding_window_

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal_,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal_,
                    window_size=(self.sliding_window_,
                                 self.sliding_window_),
                )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal_,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal_,
                    window_size=(self.sliding_window_,
                                 self.sliding_window_),
                )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:,
                                            attention_mask_num_tokens - kv_seq_len:]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(
            attention_mask)

        key_layer = index_first_axis(key_layer.reshape(
            batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(
            batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len,
                                    num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def forward(self,
                hidden_states: torch.Tensor,
                input_args: MultiLoraBatchData,
                attention_mask: Optional[torch.Tensor] = None):
        batch_size, max_seq_len, _ = hidden_states.shape

        xq = self.wq_.forward(hidden_states, input_args)
        xk = self.wk_.forward(hidden_states, input_args)
        xv = self.wv_.forward(hidden_states, input_args)

        # conver shape to multi head
        xq = xq.view(batch_size, max_seq_len, self.n_heads_,
                     self.head_dim_).transpose(1, 2)
        xk = xk.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)
        xv = xv.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)

        kv_seq_len = xk.shape[-2]

        # apply rotary embedding
        assert xq.dtype == xk.dtype
        cos = self.cos_[:max_seq_len].to(xq.dtype)
        sin = self.sin_[:max_seq_len].to(xq.dtype)
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        use_sliding_windows = (
            (self.use_sliding_window_ is None or self.use_sliding_window_)
            and (self.sliding_window_ is not None and kv_seq_len > self.sliding_window_)
            and (self.max_window_layers_ is None or self.layer_id_ < self.max_window_layers_)
        )

        # for llama2 need to repeat the heads
        # before dim: batch_size, n_kv_head, seq_len, head_dim
        # after dim: batch_size, n_head, seq_len, head_dim
        xk = repeat_kv(xk, self.n_rep_)
        xv = repeat_kv(xv, self.n_rep_)

        input_dtype = xq.dtype
        if input_dtype == torch.float32:
            if _backend.is_bf16_supported():
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float16
            xq = xq.to(target_dtype)
            xk = xk.to(target_dtype)
            xv = xv.to(target_dtype)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            xq,
            xk,
            xv,
            attention_mask,
            max_seq_len,
            use_sliding_windows=use_sliding_windows,
        ).to(input_dtype)

        attn_output = attn_output.reshape(
            batch_size, max_seq_len, self.dim_).contiguous()
        attn_output = self.wo_.forward(attn_output, input_args)

        return attn_output


LlamaAttentionClass = {
    "eager": LlamaAttention,
    "xformers": LlamaXformersAttention,
}

FlashAttentionClass = {
    "llama": LlamaFlashAttention,
    "mistral": MistralFlashAttention,
    "qwen2": MistralFlashAttention,
}


def llama_attention_factory(model_type: str, args: LLMModelArgs, **kwargs):
    if args.attn_implementation_ == "flash_attn":
        assert _is_package_available("flash_attn")
        return FlashAttentionClass[model_type](args=args, **kwargs)
    else:
        assert not args.use_sliding_window_, "Sliding window attention requires flash attention."
        return LlamaAttentionClass[args.attn_implementation_](args=args, **kwargs)
