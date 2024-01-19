from mlora.modelargs import KVCache, LoraConfig, MultiLoraBatchData

import torch
import einops

from abc import ABCMeta, abstractclassmethod
from typing import Tuple, Dict, List, Optional


def precompute_mask(tokens: torch.Tensor,
                    attention_masks: List[int],
                    n_heads: int,
                    seq_start_pos: int,
                    device: str,
                    dtype: torch.dtype) -> torch.Tensor:
    batch_size, batch_seq_len = tokens.shape

    mask = torch.full((batch_size, n_heads, batch_seq_len, batch_seq_len),
                      float("-inf"), device=device, dtype=torch.float32)
    mask = torch.triu(mask, diagonal=(seq_start_pos + 1))

    if attention_masks is not None:
        attention_masks = torch.tensor(
            attention_masks, dtype=torch.bool, device=device)
        attention_masks = attention_masks.view(batch_size, 1, 1, batch_seq_len)
        attention_masks = attention_masks.expand(-1,
                                                 n_heads, batch_seq_len, -1)
        mask = torch.masked_fill(mask, attention_masks, float("-inf"))

    mask.requires_grad_(False)
    return mask.to(device=device, dtype=dtype)


def precompute_rope_angle(dim: int, seq_len: int, device: str, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    angles = 1.0 / (theta ** (torch.arange(0, dim, 2).to(device)
                              [: (dim // 2)].to(torch.float) / dim))
    seq = torch.arange(seq_len, device=angles.device)
    emb = torch.outer(seq, angles).float()
    emb = einops.repeat(emb, "... n -> ... (n r)", r=2)

    emb.requires_grad_(False)
    # cos(angle), sin(angle)
    return (emb.cos().to(torch.float32), emb.sin().to(torch.float32))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = einops.rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return einops.rearrange(x, "... d r -> ... (d r)")


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(
        batch_size, seq_len, n_kv_heads, n_rep, head_dim).reshape(
            batch_size, seq_len, n_kv_heads * n_rep, head_dim)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     angle: Tuple[torch.Tensor, torch.Tensor],
                     dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    # data shape is: batch_size * max_seq_len * n_head * n_dim
    _, max_seq_len, _, dim_head = xq.shape

    cos = angle[0][:max_seq_len].view(max_seq_len, 1, dim_head)
    sin = angle[1][:max_seq_len].view(max_seq_len, 1, dim_head)

    xq = (xq * cos) + (rotate_half(xq) * sin)
    xk = (xk * cos) + (rotate_half(xk) * sin)
    return (xq.to(dtype), xk.to(dtype))


def apply_rotary_emb_to_one(x: torch.Tensor, angle: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    # x: batch_size, seq_len, num_head, head_dim
    _, seq_len, _, dim_head = x.shape

    cos = angle[0][:seq_len].view(seq_len, 1, dim_head)
    sin = angle[1][:seq_len].view(seq_len, 1, dim_head)

    x = (x * cos) + (rotate_half(x) * sin)
    return x


class RMSNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.norm_eps_ = eps
        self.weight_ = weight

    def _norm(self, data: torch.Tensor) -> torch.Tensor:
        return data * torch.rsqrt(+ self.norm_eps_)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        input_dtype = data.dtype
        v = data.to(torch.float32).pow(2).mean(-1, keepdim=True)
        data = data * torch.rsqrt(v + self.norm_eps_)

        return (self.weight_ * data).to(input_dtype)


class LLMModel(metaclass=ABCMeta):
    @abstractclassmethod
    def init_lora_layer_weight(self, config: LoraConfig, weight: Optional[Dict[str, torch.Tensor]]):
        pass

    @abstractclassmethod
    def load_adapter_weight(self, path: str, adapter_name: str = None):
        pass

    @abstractclassmethod
    def get_lora_weight_dict(self, lora_name: str) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        pass

    @abstractclassmethod
    def prepare_kv_cache(self, batch_size, max_seq_len) -> KVCache:
        pass

    @abstractclassmethod
    def sequential_module(self) -> torch.nn.Sequential:
        pass

    @abstractclassmethod
    def get_generate_paramas(self) -> Dict[str, any]:
        pass

    @abstractclassmethod
    def get_train_paramas(self) -> Dict[str, List[torch.Tensor]]:
        pass

    @abstractclassmethod
    def forward(self, input: MultiLoraBatchData,
                kv_cache: KVCache = None) -> torch.Tensor:
        pass
