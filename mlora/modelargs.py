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
    lora_r_: int = None
    lora_alpha_: int = None
    lora_dropout_: float = None
    target_modules_: dict = None
    prompt_template_: str = None

    def init(self, config: dict) -> "LoraConfig":
        self.lora_r_ = config["r"]
        self.lora_alpha_ = config["lora_alpha"]
        self.lora_dropout_ = config["lora_dropout"]
        self.target_modules_ = config["target_modules"]

        return self

    def export(self) -> dict:
        config = {}
        config["bias"] = "none"
        config["peft_type"] = "LORA"
        config["task_type"] = "CAUSAL_LM"
        config["r"] = self.lora_r_
        config["lora_alpha"] = self.lora_alpha_
        config["lora_dropout"] = self.lora_dropout_
        config["target_modules"] = self.target_modules_

        return config


@dataclass
class MixConfig(LoraConfig):
    # router config
    router_aux_loss_coef_: float = None
    initializer_factor_: float = None
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

    def init(self, config: dict) -> "MixConfig":
        super().init(config)
        self.router_aux_loss_coef_ = config.get(
            "router_aux_loss_coef", 0.001)  # for training
        self.initializer_factor_ = config.get(
            "initializer_factor", 1.0)  # for training
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

    def export(self) -> dict:
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


def lora_config_factory(config: dict) -> LoraConfig:
    if ("peft_type" in config and config["peft_type"] == "MIXLORA") or "routing_strategy" in config:
        return MixConfig().init(config)
    else:
        return LoraConfig().init(config)
