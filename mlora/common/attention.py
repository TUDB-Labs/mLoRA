from mlora.utils import is_package_available
from typing import Tuple, List, Optional
from .modelargs import Masks

import torch.nn.functional as F
import torch
import math

_flash_attn_available = is_package_available("flash_attn")


# input_tokens shape is: batch_size * seq_len
#   default: upper triangular matrix like below, i.e. diagonal = 1
#            0 -inf -inf
#            0    0 -inf
#            0    0    0
# additional_mask: batch_size * seq_len
#   default: is None the matrix like default, if set true, the mask metric will be -inf
#   example: [[True, False, False]]
#           -inf -inf -inf
#           -inf    0 -inf
#           -inf    0    0
def prepare_4d_causal_attention_mask(input_tokens: torch.Tensor,
                                     n_heads: int,
                                     device: str,
                                     additional_mask: List[Masks] = None,
                                     diagonal: int = 1,
                                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    batch_size, seq_len = input_tokens.shape

    TORCH_MIN_VALUE = torch.finfo(torch.float32).min
    mask = torch.full((batch_size, n_heads, seq_len, seq_len),
                      TORCH_MIN_VALUE, device=device, dtype=torch.float32)
    mask = torch.triu(mask, diagonal=diagonal)

    if additional_mask is not None:
        masks_metric = ~torch.tensor(
            additional_mask, dtype=torch.bool, device=device)
        masks_metric = masks_metric.view(batch_size, 1, 1, seq_len)
        masks_metric = masks_metric.expand(-1, n_heads, seq_len, -1)
        mask = torch.masked_fill(mask, masks_metric, TORCH_MIN_VALUE)

    mask.requires_grad_(False)

    return mask.to(device=device, dtype=dtype)


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


@torch.jit.script
def rotate_half(x: torch.Tensor):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


@torch.jit.script
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@torch.jit.script
def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, seq_len: int,
                     cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos[:seq_len].to(xq.dtype)
    sin = sin[:seq_len].to(xq.dtype)

    q_embed = (xq * cos) + (rotate_half(xq) * sin)
    k_embed = (xk * cos) + (rotate_half(xk) * sin)
    return q_embed, k_embed


def get_unpad_data(attention_mask: torch.Tensor):
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
def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                                 attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    attention_score = torch.matmul(
        query, key.transpose(2, 3)) / math.sqrt(query.size(-1))
    if attention_mask is not None:
        attention_score = attention_score + attention_mask
    attention_score = F.softmax(
        attention_score, dim=-1, dtype=torch.float32).to(value.dtype)
    attention_score = torch.matmul(attention_score, value)
    attention_score = attention_score.transpose(1, 2).contiguous()
    return attention_score
