from dataclasses import dataclass
from typing import List


@dataclass
class TokenizerArgs:
    vocab_size_: int = -1
    bos_id_: int = -1
    eos_id_: int = -1
    pad_id_: int = -1


@dataclass
class LlamaModelArgs:
    dim_: int = 4096
    multiple_of_: int = 256
    n_heads_: int = 32
    n_layers_: int = 32
    norm_eps_: float = 1e-06
    vocab_size_: int = -1
    pad_id_: int = -1
    max_seq_len_: int = 2048
    device: str = ""


@dataclass
class LoraBatchDataConfig:
    adapter_name_: str = ""
    batch_start_idx_: int = -1
    batch_end_idx_: int = -1


@dataclass
class MultiLoraBatchData:
    prompts_: List[str] = None
    lora_batch_data_config_: List[LoraBatchDataConfig] = None

    # batch seq len
    batch_seq_len_: int = None
    expand_right_: int = True

    batch_tokens_: List[List[int]] = None
    tokens_len_without_pad_: List[int] = None
