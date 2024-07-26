import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, TypeAlias, Union

import torch
from transformers.activations import ACT2FN

Tokens: TypeAlias = List[int]
Labels: TypeAlias = List[int]
Masks: TypeAlias = List[bool]


@dataclass
class Prompt:
    instruction: str = None
    input: str = None
    label: str = None


@dataclass
class InputData:
    inputs: List[Union[Prompt, List[str], str]] = None
    tokens: Optional[Tokens] = None
    labels: Optional[Labels] = None


@dataclass
class LLMModelConfig:
    name_or_path_: str = None
    device_: str = None
    dim_: int = None
    head_dim_: int = None
    intermediate_: int = None
    n_heads_: int = None
    n_kv_heads_: int = None
    n_layers_: int = None
    hidden_act_: str = None
    hidden_dropout_: float = None
    vocab_size_: int = None
    pad_token_id_: int = None
    rope_theta_: float = None
    partial_rotary_factor_: float = None
    max_seq_len_: int = None
    # eager or flash_attn
    attn_implementation_: str = "eager"
    # data type
    dtype_: torch.dtype = None


@dataclass
class LLMModelOutput:
    adapter_name: str = None
    logits: torch.Tensor = None
    loss: torch.Tensor = None
    aux_loss: torch.Tensor = None
    # for internal use
    batch_start_idx_: int = -1
    batch_end_idx_: int = -1
    loss_fn_: Callable = None


@dataclass
class LLMBatchConfig:
    adapter_name_: str = ""
    batch_start_idx_: int = -1
    batch_end_idx_: int = -1


@dataclass
class LLMModelInput:
    batch_configs_: List[LLMBatchConfig] = None
    batch_tokens_: List[Tokens] = None
    batch_labels_: List[Labels] = None
    batch_masks_: List[Masks] = None

    output_router_logits_: bool = True

    gradient_checkpoint_: str = "none"
    efficient_operator_: bool = True
    inference_mode_: bool = False


@dataclass
class AdapterConfig:
    adapter_name: str = ""
    task_name: str = "casual"

    @staticmethod
    def from_config(config: Dict[str, any]) -> "AdapterConfig":
        return AdapterConfig(
            adapter_name=config.get("name", None),
            task_name=config.get("task_name", None),
        )


lora_target_modules = {
    # LLaMA names
    "q_proj": False,
    "k_proj": False,
    "v_proj": False,
    "o_proj": False,
    "gate_proj": False,
    "down_proj": False,
    "up_proj": False,
    # Phi names
    "dense": False,
    "fc1": False,
    "fc2": False,
    # GLM names
    "qkv_proj": False,
    "dense": False,
    "dense_h_to_4h": False,
    "dense_4h_to_h": False,
}


@dataclass
class LoraConfig(AdapterConfig):
    # Weight-Decomposed Low-Rank Adaptation
    use_dora_: bool = False
    # Rank-Stabilized LoRA
    # sets the adapter scaling factor to `alpha/math.sqrt(r)`
    use_rslora_: bool = False
    # can be original or gaussian
    lora_init_: str = "original"
    lora_r_: int = None
    lora_alpha_: int = None
    lora_dropout_: float = None
    target_modules_: Dict[str, bool] = None

    def check(self) -> "LoraConfig":
        assert isinstance(self.use_dora_, bool)
        assert isinstance(self.use_rslora_, bool)
        assert isinstance(self.lora_init_, str) and self.lora_init_ in [
            "original",
            "gaussian",
        ]
        assert isinstance(self.lora_r_, int) and self.lora_r_ > 0
        assert isinstance(self.lora_alpha_, int) and self.lora_alpha_ > 0
        assert isinstance(self.lora_dropout_, float) and self.lora_dropout_ >= 0
        assert isinstance(self.target_modules_, Dict)
        for key, value in self.target_modules_.items():
            assert isinstance(key, str) and len(key) > 0
            assert isinstance(value, bool)

        return self

    @staticmethod
    def from_config(config: Dict[str, any]) -> "LoraConfig":
        lora_config = LoraConfig(**AdapterConfig.from_config(config).__dict__)
        lora_config.use_dora_ = config.get("use_dora", False)
        lora_config.use_rslora_ = config.get("use_rslora", False)
        lora_config.lora_init_ = config.get("lora_init", "original")
        lora_config.lora_r_ = config["r"]
        lora_config.lora_alpha_ = config["lora_alpha"]
        lora_config.lora_dropout_ = config["lora_dropout"]
        lora_config.target_modules_ = copy.deepcopy(lora_target_modules)
        if isinstance(config["target_modules"], List):
            for target in config["target_modules"]:
                if target in lora_target_modules:
                    lora_config.target_modules_[target] = True
        elif isinstance(config["target_modules"], Dict):
            for target, value in config["target_modules"].items():
                if target in lora_target_modules:
                    lora_config.target_modules_[target] = value
        else:
            raise ValueError("broken config item: target_modules")

        return lora_config

    def export(self) -> Dict[str, any]:
        config = {}
        if self.use_dora_:
            config["use_dora"] = True
        if self.use_rslora_:
            config["use_rslora"] = True
        config["bias"] = "none"
        config["peft_type"] = "LORA"
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
    # expert lora
    expert_config_: LoraConfig = None
    # router config
    router_aux_loss_coef_: float = None
    router_init_range_: float = None
    routing_strategy_: str = None
    jitter_noise_: float = None
    router_loss_: bool = True
    num_experts_: int = None
    act_fn_: Optional[str] = None
    # mixtral config
    top_k_: int = None
    # switch transformers config
    router_z_loss_coef_: float = None
    expert_capacity_: int = None
    ffn_dropout_: float = None
    sparse_step_: int = None

    def check(self) -> "MixConfig":
        super().check()
        if self.expert_config_ is not None:
            self.expert_config_.check()
        assert (
            isinstance(self.router_aux_loss_coef_, float)
            and self.router_aux_loss_coef_ >= 0
        )
        assert (
            isinstance(self.router_init_range_, float) and self.router_init_range_ >= 0
        )
        assert (
            isinstance(self.routing_strategy_, str)
            and self.routing_strategy_ in available_routing_strategies
        )
        assert isinstance(self.jitter_noise_, float) and self.jitter_noise_ >= 0
        assert isinstance(self.router_loss_, bool)
        assert isinstance(self.num_experts_, int) and self.num_experts_ > 0
        assert self.act_fn_ is None or (
            isinstance(self.act_fn_, str) and self.act_fn_ in ACT2FN
        )
        if self.routing_strategy_ == "mixtral":
            assert isinstance(self.top_k_, int) and self.top_k_ > 0
        elif self.routing_strategy_ == "switch":
            assert (
                isinstance(self.router_z_loss_coef_, float)
                and self.router_z_loss_coef_ >= 0
            )
            if self.sparse_step_ is not None:
                assert isinstance(self.sparse_step_, int) and self.sparse_step_ > 0
            assert isinstance(self.expert_capacity_, int) and self.expert_capacity_ > 0
            assert isinstance(self.ffn_dropout_, float) and self.ffn_dropout_ >= 0

        return self

    @staticmethod
    def from_config(config: Dict[str, any]) -> "MixConfig":
        lora_config = MixConfig(**LoraConfig.from_config(config).__dict__)
        if "expert_lora" in config:
            expert_config = copy.deepcopy(config)
            expert_config.update(config["expert_lora"])
            lora_config.expert_config_ = LoraConfig().from_config(expert_config)
        lora_config.router_aux_loss_coef_ = config.get(
            "router_aux_loss_coef", 0.001
        )  # for training
        lora_config.routing_strategy_ = config["routing_strategy"]
        lora_config.router_loss_ = config.get("router_loss", True)
        lora_config.num_experts_ = config["num_experts"]
        # silu for mixtral or gelu_new for switch transformers
        # left blank to automatically use the original act_fn of FFN
        lora_config.act_fn_ = config.get("act_fn", None)
        if lora_config.routing_strategy_ == "mixtral":
            lora_config.router_init_range_ = config.get("router_init_range", 0.02)
            lora_config.jitter_noise_ = config.get("jitter_noise", 0.0)
            lora_config.top_k_ = config.get("top_k", 2)
        elif lora_config.routing_strategy_ == "switch":
            lora_config.router_init_range_ = config.get("router_init_range", 1.0)
            lora_config.jitter_noise_ = config.get("jitter_noise", 0.01)
            lora_config.router_z_loss_coef_ = config.get(
                "router_z_loss_coef", 0.001
            )  # for training
            # expert_capacity = (max_sequence_length / num_experts) * capacity_factor
            # common values of capacity_factor: 1.0, 1.25, 2.0
            lora_config.expert_capacity_ = config.get("expert_capacity", 32)
            lora_config.ffn_dropout_ = config.get("ffn_dropout", 0.0)
            lora_config.sparse_step_ = config.get("sparse_step", None)

        return lora_config

    def export(self) -> Dict[str, any]:
        config = super().export()
        config["peft_type"] = "MIXLORA"
        if self.expert_config_ is not None:
            expert_config = self.expert_config_.export()
            expert_config.pop("peft_type")
            expert_config.pop("target_modules")
            config["expert_lora"] = expert_config
        config["routing_strategy"] = self.routing_strategy_
        config["num_experts"] = self.num_experts_
        if self.act_fn_ is not None:
            config["act_fn"] = self.act_fn_
        if self.routing_strategy_ == "mixtral":
            config["top_k"] = self.top_k_
        elif self.routing_strategy_ == "switch":
            config["expert_capacity"] = self.expert_capacity_
            config["sparse_step"] = self.sparse_step_

        return config

    def expert_config(self, expert_idx: int) -> LoraConfig:
        if self.expert_config_ is None:
            config = copy.deepcopy(super())
        else:
            config = copy.deepcopy(self.expert_config_)
        config.adapter_name = f"moe.{self.adapter_name}.experts.{expert_idx}"
        return config


def lora_config_factory(config: Dict[str, any]) -> LoraConfig:
    if (
        "peft_type" in config and config["peft_type"] == "MIXLORA"
    ) or "routing_strategy" in config:
        return MixConfig.from_config(config).check()
    else:
        return LoraConfig.from_config(config).check()
