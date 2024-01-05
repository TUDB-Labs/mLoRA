from mlora.modelargs import LoraConfig, MultiLoraBatchData

import torch
import einops

from abc import ABCMeta, abstractclassmethod
from typing import Tuple, Dict, List, Optional


def precompute_mask(input: MultiLoraBatchData,
                    n_head: int,
                    device: str,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    mask = torch.full((len(input.prompts_), n_head,
                      input.batch_seq_len_, input.batch_seq_len_), float("-inf"))
    mask = torch.triu(mask, diagonal=1).to(torch.float32).cuda(device)

    for idx, _ in enumerate(input.prompts_):
        zero_len = input.tokens_len_without_pad_[idx]
        inf_len = input.batch_seq_len_ - zero_len
        expand_side = input.expand_side_[idx]

        if expand_side == "right":
            mask[idx] += torch.tensor([0] * zero_len + [float("-inf")] * inf_len).expand(
                input.batch_seq_len_, input.batch_seq_len_).cuda(device)
        else:
            mask[idx] += torch.tensor([float("-inf")] * inf_len + [0] * zero_len).expand(
                input.batch_seq_len_, input.batch_seq_len_).cuda(device)

    mask.requires_grad_(False)
    return mask.to(dtype)


def precompute_mask_for_inference(input: MultiLoraBatchData,
                                  seq_pos: int,
                                  device: str,
                                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    mask = torch.full((1, 1, input.batch_seq_len_,
                      input.batch_seq_len_), float("-inf"), device=device)
    mask = mask.to(torch.float32).triu(diagonal=(seq_pos + 1))
    return mask.to(dtype)


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


class KVCache:
    def __init__(self) -> None:
        self.cache_k: List[torch.Tensor] = []
        self.cache_v: List[torch.Tensor] = []
        self.seq_pos: int = 0

    def update(self, xk: torch.Tensor, xv: torch.Tensor, layer_idx: int,
               bsz: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.cache_k) <= layer_idx:
            self.cache_k.append(xk)
            self.cache_v.append(xv)
        else:
            self.cache_k[layer_idx][:bsz,
                                    self.seq_pos: self.seq_pos + seq_len] = xk
            self.cache_v[layer_idx][:bsz,
                                    self.seq_pos: self.seq_pos + seq_len] = xv

        return self.cache_k[layer_idx][:bsz, :self.seq_pos + seq_len], \
            self.cache_v[layer_idx][:bsz, :self.seq_pos + seq_len]


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
    def save_adapter_weight(self, path: str, dir_suffix=""):
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
                output_router_logits: bool = False,
                kv_cache: KVCache = None) -> torch.Tensor:
        pass
