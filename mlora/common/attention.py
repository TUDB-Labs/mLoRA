import inspect
import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from transformers.utils import is_flash_attn_2_available

from .cache import Cache, StaticCache

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(
        inspect.signature(flash_attn_func).parameters
    )


def prepare_4d_causal_attention_mask(
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
) -> torch.Tensor:
    past_seen_tokens = (
        past_key_values.get_seq_length() if past_key_values is not None else 0
    )

    if past_seen_tokens is None:
        past_seen_tokens = 0

    using_static_cache = isinstance(past_key_values, StaticCache)

    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    if using_static_cache:
        target_length = past_key_values.get_max_length()
    else:
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

    causal_mask = torch.full(
        (sequence_length, target_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(
        -1, 1
    )
    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    if attention_mask is not None:
        causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        mask_length = attention_mask.shape[-1]
        padding_mask = (
            causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        )
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[
            :, :, :, :mask_length
        ].masked_fill(padding_mask, min_dtype)

    return causal_mask


def eager_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attention_score = torch.matmul(
        query_states, key_states.transpose(2, 3)
    ) / math.sqrt(query_states.size(-1))
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attention_score = attention_score + causal_mask
    attention_score = F.softmax(attention_score, dim=-1, dtype=torch.float32).to(
        value_states.dtype
    )
    attention_score = torch.matmul(attention_score, value_states)
    attention_score = attention_score.transpose(1, 2).contiguous()
    return attention_score


def _get_unpad_data(
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _upad_input(
    query_layer: torch.Tensor,
    key_layer: torch.Tensor,
    value_layer: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
        indices_k,
    )
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
        indices_k,
    )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(
            query_layer.reshape(batch_size * kv_seq_len, -1, head_dim), indices_k
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
            query_layer, attention_mask
        )

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
):
    if not use_top_left_mask:
        causal = is_causal
    else:
        causal = is_causal and query_length != 1

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        _flash_supports_window_size
        and sliding_window is not None
        and key_states.shape[1] > sliding_window
    )
    flash_kwargs = (
        {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}
    )

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = (
            _upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )
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
            causal=causal,
            **flash_kwargs,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    else:
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )

    return attn_output
