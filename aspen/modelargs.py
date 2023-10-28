from dataclasses import dataclass
from typing import List

Tokens = List[int]


@dataclass
class TokenizerArgs:
    vocab_size_: int = -1
    bos_id_: int = -1
    eos_id_: int = -1
    pad_id_: int = -1


@dataclass
class LLMModelArgs:
    dim_: int = 4096
    multiple_of_: int = 256
    n_heads_: int = 32
    n_kv_heads_: int = 32
    n_layers_: int = 32
    norm_eps_: float = 1e-06
    hidden_dropout_: float = 0.0
    vocab_size_: int = -1
    pad_token_id_: int = -1
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
    # the expand right and tokens without pad len
    # be need by the mask matrix
    batch_seq_len_: int = None
    expand_right_: int = True

    batch_tokens_: List[Tokens] = None
    tokens_len_without_pad_: Tokens = None

    inference_model_: bool = False
