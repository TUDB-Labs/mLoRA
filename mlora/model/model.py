from mlora.model.modelargs import MultiLoraBatchData, Masks
from mlora.config import LoraConfig

import torch

from abc import ABCMeta, abstractclassmethod
from typing import Tuple, Dict, List, Optional


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
def precompute_mask(input_tokens: torch.Tensor,
                    n_heads: int,
                    device: str,
                    additional_mask: List[Masks] = None,
                    diagonal: int = 1,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if input_tokens.dim() == 2:
        batch_size, seq_len = input_tokens.shape
    elif input_tokens.dim() == 3:
        batch_size, seq_len, _ = input_tokens.shape
    else:
        raise Exception("input dim is not correct {input_tokens.dim}")

    TORCH_MIN_VALUE = torch.finfo(dtype).min
    mask = torch.full((batch_size, n_heads, seq_len, seq_len),
                      TORCH_MIN_VALUE, device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=diagonal)

    if additional_mask is not None:
        masks_metric = torch.tensor(
            additional_mask, dtype=torch.bool, device=device)
        masks_metric = masks_metric.view(batch_size, 1, 1, seq_len)
        masks_metric = masks_metric.expand(-1, n_heads, seq_len, -1)
        mask = torch.masked_fill(mask, masks_metric, TORCH_MIN_VALUE)

    mask.requires_grad_(False)

    return mask.to(device=device, dtype=dtype)


def precompute_rope_angle(dim: int, seq_len: int, theta: float, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # this implement is different with facebooksearch/llama
    #   ref: https://github.com/huggingface/transformers/issues/25199
    angles = 1.0 / \
        (theta ** (torch.arange(0, dim, 2).float().to(device) / dim))
    seq = torch.arange(seq_len, device=device, dtype=angles.dtype)
    emb = torch.outer(seq, angles)
    emb = torch.cat((emb, emb), dim=-1)

    emb.requires_grad_(False)
    # cos(angle), sin(angle)
    return (emb.cos(), emb.sin())


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    # see the above ref
    left_part = x[..., :x.shape[-1] // 2]
    right_part = x[..., x.shape[-1] // 2:]
    return torch.cat((-right_part, left_part), dim=-1)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # data shape is: batch_size * n_head * seq_len * n_dim
    xq_embed = (xq * cos) + (rotate_half(xq) * sin)
    xk_embed = (xk * cos) + (rotate_half(xk) * sin)
    return (xq_embed, xk_embed)


def apply_rotary_emb_to_one(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: batch_size, seq_len, num_head, head_dim
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, n_kv_heads, seq_len, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(
        batch, n_kv_heads, n_rep, seq_len, head_dim)
    x = x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)
    return x


class LLMModel(metaclass=ABCMeta):
    vocab_size_: int = -1
    n_heads_: int = -1

    @abstractclassmethod
    def forward(self, input: MultiLoraBatchData):
        pass

    @abstractclassmethod
    def get_train_paramas(self) -> Dict[str, List[torch.Tensor]]:
        pass

    @abstractclassmethod
    def init_lora_weight(self, lora_config: LoraConfig, weight: Optional[Dict[str, torch.Tensor]]):
        pass

    @abstractclassmethod
    def get_lora_weight_dict(self, lora_name: str) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        pass

    @abstractclassmethod
    def sequential_module(self) -> torch.nn.Sequential:
        pass
