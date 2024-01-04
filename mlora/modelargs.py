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
    expand_side_: List[str] = None

    batch_tokens_: List[Tokens] = None
    tokens_len_without_pad_: Tokens = None

    inference_model_: bool = False


@dataclass
class LoraConfig:
    adapter_name_: str = ""
    device_: str = "cuda:0"
    lora_r_: int = 8
    lora_alpha_: int = 16
    lora_dropout_: float = 0.05
    target_modules_: dict = None


@dataclass
class MixConfig(LoraConfig):
    # router config
    router_aux_loss_coef_: float = 0.001
    initializer_factor_: float = 1.0
    routing_strategy_: str = "basic"
    num_experts_: int = 8
    # default, silu or gelu_new
    act_fn_: str = "default"
    # for top-k moes
    top_k_: int = 2
    # for switch transformers
    router_z_loss_coef_: float = 0.001
    # expert_capacity = (max_sequence_length / num_experts) * capacity_factor
    # common values of capacity_factor: 1.0, 1.25, 2.0
    expert_capacity_: int = 64
    sparse_layers_: int = 16
    jitter_noise_: float = 0.1
    dropout_rate_: float = 0.1
