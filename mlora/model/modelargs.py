from dataclasses import dataclass
from typing import List

from transformers import PretrainedConfig
import logging
import torch

Tokens = List[int]
Masks = List[bool]


@dataclass
class TokenizerArgs:
    vocab_size_: int = -1
    bos_id_: int = -1
    eos_id_: int = -1
    pad_id_: int = -1


@dataclass
class LLMModelArgs:
    name_or_path_: str = ""
    dim_: int = 4096
    multiple_of_: int = 256
    n_heads_: int = 32
    n_kv_heads_: int = 32
    n_layers_: int = 32
    rope_theta_: float = 10000.0
    norm_eps_: float = 1e-06
    hidden_dropout_: float = 0.0
    vocab_size_: int = -1
    pad_token_id_: int = -1
    max_seq_len_: int = 4096
    device_: str = ""
    dtype_: torch.dtype = None

    def __init__(self, config: PretrainedConfig):
        self.from_pretrained_config(config)

    def from_pretrained_config(self, config: PretrainedConfig):
        self.name_or_path_ = config.name_or_path
        self.dim_ = config.hidden_size
        self.n_heads_ = config.num_attention_heads
        if hasattr(config, "num_key_value_heads"):
            self.n_kv_heads_ = config.num_key_value_heads
        self.n_layers_ = config.num_hidden_layers
        self.norm_eps_ = config.rms_norm_eps
        self.vocab_size_ = config.vocab_size
        self.pad_token_id_ = config.pad_token_id
        if hasattr(config, "max_sequence_length"):
            self.max_seq_len_ = config.max_sequence_length
        if hasattr(config, "sliding_window") and self.max_seq_len_ > config.sliding_window:
            logging.warning(
                "Shrink max sequence length to window size of sliding window attention.")
            self.max_seq_len_ = config.sliding_window
        if hasattr(config, "rope_theta"):
            self.rope_theta_ = config.rope_theta


@dataclass
class LoraBatchDataConfig:
    adapter_name_: str = ""
    batch_start_idx_: int = -1
    batch_end_idx_: int = -1


@dataclass
class MultiLoraBatchData:
    # the datas: batch_size * token
    batch_tokens_: List[Tokens] = None
    additional_mask_: List[Masks] = None
    lora_batch_data_config_: List[LoraBatchDataConfig] = None

    inference_model_: bool = False
