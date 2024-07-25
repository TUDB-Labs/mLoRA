import math
from typing import Optional, Tuple

import torch

from .modelargs import LLMModelConfig


def _compute_default_rope_parameters(
    config: Optional[LLMModelConfig] = None,
    device: Optional[torch.device] = None,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple[torch.Tensor, float]:
    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        base = config.rope_theta_
        partial_rotary_factor = (
            config.partial_rotary_factor_
            if config.partial_rotary_factor_ is not None
            else 1.0
        )
        dim = int((config.dim_ // config.n_heads_) * partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (
        base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
    )
    return inv_freq, attention_factor


def _compute_llama3_parameters(
    config: LLMModelConfig,
    device: torch.device,
    seq_len: Optional[int] = None,
    **rope_kwargs,
) -> Tuple[torch.Tensor, float]:
    # Gets the default RoPE parameters
    inv_freq, attention_factor = _compute_default_rope_parameters(
        config, device, seq_len, **rope_kwargs
    )

    factor = config.rope_scaling_["factor"]  # `8` in the original implementation
    low_freq_factor = config.rope_scaling_[
        "low_freq_factor"
    ]  # `1` in the original implementation
    high_freq_factor = config.rope_scaling_[
        "high_freq_factor"
    ]  # `4` in the original implementation
    old_context_len = config.rope_scaling_[
        "original_max_position_embeddings"
    ]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in inv_freq:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    inv_freq = torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)
    return inv_freq, attention_factor


ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "llama3": _compute_llama3_parameters,
}
