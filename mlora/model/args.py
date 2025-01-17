import logging
import uuid
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
from transformers import PretrainedConfig

Tokens = List[int]
Masks = List[bool]


@dataclass
class LLMModelArgs:
    name_or_path_: str
    dim_: int
    multiple_of_: int
    n_heads_: int
    n_kv_heads_: int
    n_layers_: int
    rope_theta_: float
    norm_eps_: float
    hidden_dropout_: float
    vocab_size_: int
    pad_token_id_: int
    max_seq_len_: int
    device_: str
    dtype_: torch.dtype

    def __init__(self, config: PretrainedConfig):
        self.__from_pretrained_config(config)

    def __from_pretrained_config(self, config: PretrainedConfig):
        self.name_or_path_ = config.name_or_path
        self.dim_ = config.hidden_size
        self.multiple_of_ = 256
        self.n_heads_ = config.num_attention_heads
        if hasattr(config, "num_key_value_heads"):
            self.n_kv_heads_ = config.num_key_value_heads
        self.n_layers_ = config.num_hidden_layers
        self.rope_theta_ = 10000.0
        self.norm_eps_ = config.rms_norm_eps
        self.hidden_dropout_ = 0.0
        self.vocab_size_ = config.vocab_size
        self.pad_token_id_ = config.pad_token_id
        self.max_seq_len_ = 4096
        if hasattr(config, "max_sequence_length"):
            self.max_seq_len_ = config.max_sequence_length

        if (
            hasattr(config, "sliding_window")
            and self.max_seq_len_ > config.sliding_window
        ):
            logging.warning(
                "Shrink max sequence length to window size of sliding window attention."
            )
            self.max_seq_len_ = config.sliding_window

        if hasattr(config, "rope_theta"):
            self.rope_theta_ = config.rope_theta

        self.device_ = ""
        self.dtype_ = torch.float32


@dataclass
class LinearInfo:
    name_: str
    in_dim_: int
    out_dim_: int
    base_weight_: torch.nn.Linear


@dataclass
class ModelDataConfig:
    adapter_name_: str
    adapter_type_: str

    batch_start_idx_: int
    batch_end_idx_: int


@dataclass
class ModelData:
    batch_tokens_: List[Tokens]
    batch_mask_: List[Masks]
    data_config_: List[ModelDataConfig]

    enable_checkpoint_: bool

    # the flag for serialize
    random_id_: int
    task_name_: List[str]


class MLoRADataConfig:
    adapter_name_: str
    adapter_type_: str

    batch_start_idx_: int
    batch_end_idx_: int

    expand_fn_: Callable[
        [List[Tokens], Optional[int]], Tuple[List[Tokens], List[Masks]]
    ]
    loss_fn_: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor], Optional[torch.Tensor]
    ]

    task_name_: str

    def __init__(
        self,
        adapter_name: str,
        adapter_type: str,
        start_idx: int,
        end_idx: int,
        expand_fn: Callable,
        loss_fn: Callable,
        task_name: str,
    ) -> None:
        self.adapter_name_ = adapter_name
        self.adapter_type_ = adapter_type
        self.batch_start_idx_ = start_idx
        self.batch_end_idx_ = end_idx

        self.expand_fn_ = expand_fn
        self.loss_fn_ = loss_fn

        self.task_name_ = task_name

    def model_data_config(self) -> ModelDataConfig:
        return ModelDataConfig(
            adapter_name_=self.adapter_name_,
            adapter_type_=self.adapter_type_,
            batch_start_idx_=self.batch_start_idx_,
            batch_end_idx_=self.batch_end_idx_,
        )


class MLoRAData:
    # the datas: batch_size * token
    batch_tokens_: List[Tokens]
    batch_mask_: List[Masks]
    data_config_: List[MLoRADataConfig]

    # the flag for serialize
    random_id_: int

    def __init__(
        self,
        batch_tokens: List[Tokens],
        batch_mask: List[Masks],
        data_config: List[MLoRADataConfig],
    ) -> None:
        self.batch_tokens_ = batch_tokens
        self.batch_mask_ = batch_mask
        self.data_config_ = data_config
        self.random_id_ = uuid.uuid4().int

    def model_data(self) -> ModelData:
        return ModelData(
            batch_tokens_=self.batch_tokens_,
            batch_mask_=self.batch_mask_,
            data_config_=[config.model_data_config() for config in self.data_config_],
            enable_checkpoint_=True,
            task_name_=[config.task_name_ for config in self.data_config_],
            random_id_=self.random_id_,
        )

    def batch_size(self) -> int:
        return len(self.batch_tokens_)

    def token_len(self) -> int:
        return len(self.batch_tokens_[0])
