from mlora.modelargs import MultiLoraBatchData, Masks

import torch
import einops

from abc import ABCMeta, abstractclassmethod
from typing import Tuple, Dict, List, Optional


# input_tokens shape is: batch_size * seq_len
#   default: upper triangular matrix like below, i.e. diagonal = 1
#            0 -inf -inf
#            0    0 -inf
#            0    0    0
# additional_mask: batch_size * seq_len
#   default: is None, if set true, the mask metric will be -inf
#   example: [[True, True, False]]
#           -inf -inf -inf
#           -inf    0    0
#           -inf    0    0
def precompute_mask(input_tokens: torch.Tensor,
                    n_heads: int,
                    device: str,
                    additional_mask: List[Masks] = None,
                    diagonal: int = 1,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    batch_size, seq_len = input_tokens.shape

    mask = torch.full((batch_size, n_heads, seq_len, seq_len),
                      float("-inf"), device=device, dtype=dtype)
    mask = torch.triu(mask, diagonal=diagonal)

    if additional_mask is not None:
        masks_metric = torch.tensor(
            additional_mask, dtype=torch.bool, device=device)
        masks_metric = masks_metric.view(batch_size, 1, 1, seq_len)
        masks_metric = masks_metric.expand(-1, n_heads, seq_len, -1)
        mask = torch.masked_fill(mask, masks_metric, float("-inf"))

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
                     angle: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    # data shape is: batch_size * max_seq_len * n_head * n_dim
    _, max_seq_len, _, dim_head = xq.shape

    cos = angle[0][:max_seq_len].view(max_seq_len, 1, dim_head)
    sin = angle[1][:max_seq_len].view(max_seq_len, 1, dim_head)

    xq = (xq * cos) + (rotate_half(xq) * sin)
    xk = (xk * cos) + (rotate_half(xk) * sin)
    return (xq, xk)


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
    def forward(self, input: MultiLoraBatchData):
        pass

    @abstractclassmethod
    def get_train_paramas(self) -> Dict[str, List[torch.Tensor]]:
        pass

    @abstractclassmethod
    def init_lora_weight(self,
                         adapter_name: str,
                         r: int,
                         lora_alpha: int,
                         lora_dropout: float,
                         target: Dict[str, bool],
                         weight: Optional[Dict[str, torch.Tensor]]):
        pass

    @abstractclassmethod
    def get_lora_weight_dict(self, lora_name: str) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        pass

    @abstractclassmethod
    def sequential_module(self) -> torch.nn.Sequential:
        pass
