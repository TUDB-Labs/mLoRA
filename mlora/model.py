from mlora.modelargs import LoraConfig, MultiLoraBatchData, LLMModelOutput

import torch

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
                      torch.finfo(dtype).min, device=device, dtype=torch.float32)
    mask = torch.triu(mask, diagonal=(seq_start_pos + 1))

    if attention_masks is not None:
        attention_masks = torch.tensor(
            attention_masks, dtype=torch.bool, device=device)
        attention_masks = attention_masks.view(batch_size, 1, 1, batch_seq_len)
        attention_masks = attention_masks.expand(-1,
                                                 n_heads, batch_seq_len, -1)
        mask = torch.masked_fill(mask, attention_masks, torch.finfo(dtype).min)

    mask.requires_grad_(False)
    return mask.to(device=device, dtype=dtype)


def precompute_rope_angle(dim: int, seq_len: int,
                          theta: float = 10000.0,
                          device: str = None) -> Tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / \
        (theta ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
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
        self.variance_epsilon_ = eps
        self.weight_ = weight

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * \
            torch.rsqrt(variance + self.variance_epsilon_)
        return self.weight_ * hidden_states.to(input_dtype)


class LLMOutput(metaclass=ABCMeta):
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        pass

    def loss(self,
             input_ids: torch.Tensor,
             output_logits: torch.Tensor,
             labels: List[List[int]]) -> torch.Tensor:
        pass

    def state_dict(self):
        return {}


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
                labels: List[List[int]] = None) -> List[LLMModelOutput]:
        pass
