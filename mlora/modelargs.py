from transformers.activations import ACT2FN
from typing import List, Dict, Callable
from dataclasses import dataclass

import torch


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
    rope_theta_: float = 10000.0
    max_seq_len_: int = 2048
    device_: str = ""
    dtype_: torch.dtype = None


@dataclass
class LLMModelOutput:
    adapter_name: str = None
    logits: torch.Tensor = None
    loss: torch.Tensor = None
    # for internal use
    batch_start_idx_: int = -1
    batch_end_idx_: int = -1
    loss_fn_: Callable = None


@dataclass
class LoraBatchDataConfig:
    adapter_name_: str = ""
    batch_start_idx_: int = -1
    batch_end_idx_: int = -1


@dataclass
class MultiLoraBatchData:
    lora_batch_data_config_: List[LoraBatchDataConfig] = None

    batch_tokens_: List[Tokens] = None
    attention_masks_: List[Tokens] = None

    gradient_checkpoint_: bool = True
    inference_seq_pos_: int = -1

    @property
    def inference_mode_(self) -> bool:
        return self.inference_seq_pos_ >= 0


@dataclass
class LoraConfig:
    adapter_name_: str = ""
    task_type_: str = "casual"
    device_: str = "cuda:0"
    dtype_: torch.dtype = None
    lora_r_: int = None
    lora_alpha_: int = None
    lora_dropout_: float = None
    target_modules_: Dict[str, bool] = None

    def check(self) -> "LoraConfig":
        assert isinstance(self.lora_r_, int) and self.lora_r_ > 0
        assert isinstance(self.lora_alpha_, int) and self.lora_alpha_ > 0
        assert isinstance(self.lora_dropout_,
                          float) and self.lora_dropout_ >= 0
        assert isinstance(self.target_modules_, Dict)
        for key, value in self.target_modules_.items():
            assert isinstance(key, str) and len(key) > 0
            assert isinstance(value, bool)

        return self

    def from_config(self, config: Dict[str, any]) -> "LoraConfig":
        self.lora_r_ = config["r"]
        self.lora_alpha_ = config["lora_alpha"]
        self.lora_dropout_ = config["lora_dropout"]
        self.target_modules_ = {
            # LLaMA names
            "q_proj": False,
            "k_proj": False,
            "v_proj": False,
            "o_proj": False,
            "w1_proj": False,
            "w2_proj": False,
            "w3_proj": False,
            # ChatGLM names
            "qkv": False,
            "dense": False,
            "mlp_in": False,
            "mlp_out": False
        }
        if isinstance(config["target_modules"], List):
            for target in config["target_modules"]:
                if target in self.target_modules_:
                    self.target_modules_[target] = True
        elif isinstance(config["target_modules"], Dict):
            for target, value in config["target_modules"].items():
                if target in self.target_modules_:
                    self.target_modules_[target] = value
        else:
            raise ValueError("broken config item: target_modules")

        return self

    def export(self) -> Dict[str, any]:
        config = {}
        config["bias"] = "none"
        config["peft_type"] = "LORA"
        config["task_type"] = "CAUSAL_LM"
        config["r"] = self.lora_r_
        config["lora_alpha"] = self.lora_alpha_
        config["lora_dropout"] = self.lora_dropout_
        tgt_list = []
        for target, value in self.target_modules_.items():
            if value:
                tgt_list.append(target)
        config["target_modules"] = tgt_list

        return config


available_routing_strategies = ["mixtral", "switch"]


@dataclass
class MixConfig(LoraConfig):
    # router config
    router_aux_loss_coef_: float = None
    routing_strategy_: str = None
    num_experts_: int = None
    act_fn_: str = None
    # mixtral config
    top_k_: int = None
    # switch transformers config
    router_z_loss_coef_: float = None
    expert_capacity_: int = None
    jitter_noise_: float = None
    dropout_rate_: float = None

    def check(self) -> "MixConfig":
        super().check()
        assert isinstance(self.router_aux_loss_coef_,
                          float) and self.router_aux_loss_coef_ >= 0
        assert isinstance(self.routing_strategy_,
                          str) and self.routing_strategy_ in available_routing_strategies
        assert isinstance(self.num_experts_, int) and self.num_experts_ > 0
        assert isinstance(self.act_fn_, str) and self.act_fn_ in ACT2FN
        if self.routing_strategy_ == "mixtral":
            assert isinstance(self.top_k_, int) and self.top_k_ > 0
        elif self.routing_strategy_ == "switch":
            assert isinstance(self.router_z_loss_coef_,
                              float) and self.router_z_loss_coef_ >= 0
            assert isinstance(self.expert_capacity_,
                              int) and self.expert_capacity_ > 0
            assert isinstance(self.jitter_noise_,
                              float) and self.jitter_noise_ >= 0
            assert isinstance(self.dropout_rate_,
                              float) and self.dropout_rate_ >= 0

        return self

    def from_config(self, config: Dict[str, any]) -> "MixConfig":
        super().from_config(config)
        self.router_aux_loss_coef_ = config.get(
            "router_aux_loss_coef", 0.001)  # for training
        self.routing_strategy_ = config["routing_strategy"]
        self.num_experts_ = config["num_experts"]
        # silu for mixtral or gelu_new for switch transformers
        self.act_fn_ = config["act_fn"]
        if self.routing_strategy_ == "mixtral":
            self.top_k_ = config.get("top_k", 2)
        elif self.routing_strategy_ == "switch":
            self.router_z_loss_coef_ = config.get(
                "router_z_loss_coef", 0.001)  # for training
            # expert_capacity = (max_sequence_length / num_experts) * capacity_factor
            # common values of capacity_factor: 1.0, 1.25, 2.0
            self.expert_capacity_ = config.get("expert_capacity", 64)
            self.jitter_noise_ = config.get("jitter_noise", 0.1)
            self.dropout_rate_ = config.get("ffn_dropout", 0.1)

        return self

    def export(self) -> Dict[str, any]:
        config = super().export()
        config["peft_type"] = "MIXLORA"
        config["routing_strategy"] = self.routing_strategy_
        config["num_experts"] = self.num_experts_
        config["act_fn"] = self.act_fn_
        if self.routing_strategy_ == "mixtral":
            config["top_k"] = self.top_k_
        elif self.routing_strategy_ == "switch":
            config["expert_capacity"] = self.expert_capacity_
            config["jitter_noise"] = self.jitter_noise_
            config["ffn_dropout"] = self.dropout_rate_

        return config


def lora_config_factory(config: Dict[str, any]) -> LoraConfig:
    if ("peft_type" in config and config["peft_type"] == "MIXLORA") or "routing_strategy" in config:
        return MixConfig().from_config(config).check()
    else:
        return LoraConfig().from_config(config).check()
