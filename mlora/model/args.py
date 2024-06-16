import torch
import logging
from typing import List, Callable, Optional, Tuple
from dataclasses import dataclass
from transformers import PretrainedConfig

Tokens = List[int]
Masks = List[bool]


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
        self.__from_pretrained_config(config)

    def __from_pretrained_config(self, config: PretrainedConfig):
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
class LinearInfo:
    name_: str
    in_dim_: int
    out_dim_: int


@dataclass
class ModelDataConfig:
    adapter_name_: str = ""
    adapter_type_: str = ""

    batch_start_idx_: int = -1
    batch_end_idx_: int = -1


@dataclass
class ModelData:
    batch_tokens_: List[Tokens] = None
    batch_mask_: List[Masks] = None
    data_config_: List[ModelDataConfig] = None

    enable_checkpoint_: bool = True


class MLoRADataConfig:
    adapter_name_: str = ""
    adapter_type_: str = ""

    batch_start_idx_: int = -1
    batch_end_idx_: int = -1

    expand_fn_: Callable[[List[Tokens], Optional[int]],
                         Tuple[List[Tokens], List[Masks]]] = None
    loss_fn_: Callable[[torch.Tensor, torch.Tensor,
                        torch.Tensor], torch.Tensor] = None

    def __init__(self, adapter_name: str, adapter_type: str,
                 start_idx: int, end_idx: int,
                 expand_fn: Callable, loss_fn: Callable) -> None:
        self.adapter_name_ = adapter_name
        self.adapter_type_ = adapter_type
        self.batch_start_idx_ = start_idx
        self.batch_end_idx_ = end_idx

        self.expand_fn_ = expand_fn
        self.loss_fn_ = loss_fn

    def model_data_config(self) -> ModelDataConfig:
        return ModelDataConfig(adapter_name_=self.adapter_name_,
                               adapter_type_=self.adapter_type_,
                               batch_start_idx_=self.batch_start_idx_,
                               batch_end_idx_=self.batch_end_idx_)


class MLoRAData:
    # the datas: batch_size * token
    batch_tokens_: List[Tokens] = None
    batch_mask_: List[Masks] = None
    data_config_: List[MLoRADataConfig] = None

    def __init__(self,
                 batch_tokens: List[Tokens],
                 batch_mask: List[Masks],
                 data_config: List[MLoRADataConfig]) -> None:
        self.batch_tokens_ = batch_tokens
        self.batch_mask_ = batch_mask
        self.data_config_ = data_config

    def model_data(self) -> ModelData:
        return ModelData(batch_tokens_=self.batch_tokens_,
                         batch_mask_=self.batch_mask_,
                         data_config_=[config.model_data_config() for config in self.data_config_],
                         enable_checkpoint_=True)
