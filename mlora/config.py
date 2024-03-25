import json

from typing import Dict, List


class DictConfig:
    __params_map: Dict[str, str] = {}

    def __init__(self, config: Dict[str, str]) -> None:
        self.init(self.__params_map, config)

    def init(self,
             params_map: Dict[str, str],
             config: Dict[str, str]):
        for key in params_map:
            setattr(self, key, config[params_map[key]])


class OptimConfig(DictConfig):
    optim_: str = ""
    lr_: float = 0.0

    __params_map: Dict[str, str] = {
        "lr_": "lr",
        "optim_": "optim"
    }

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)


class SGDOptimConfig(OptimConfig):
    momentum_: float = 0.0

    __params_map: Dict[str, str] = {
        "momentum_": "momentum"
    }

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)


class AdamWOptimConfig(OptimConfig):
    __params_map: Dict[str, str] = {}

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)


class LoraConfig(DictConfig):
    adapter_name_: str = ""
    r_: int = -1
    lora_alpha_: int = 0
    lora_dropout_: float = 0.05
    target_: Dict[str, bool] = {}

    batch_size_: int = 0
    micro_batch_size_: int = 0
    test_batch_size_: int = 0
    num_epochs_: int = 0

    data_: str = ""
    test_data_: str = ""
    prompt_: str = ""

    group_by_length_: bool = False
    expand_side_: str = "right"
    expand_token_id_: int = 0

    optim_config_: OptimConfig = None

    val_set_size_: int = -1
    cutoff_len_: int = -1

    __params_map: Dict[str, str] = {
        "adapter_name_": "name",
        "r_": "r",
        "lora_alpha_": "alpha",
        "lora_dropout_": "dropout",
        "target_": "target_modules",

        "batch_size_": "batch_size",
        "micro_batch_size_": "micro_batch_size",
        "test_batch_size_": "test_batch_size",
        "num_epochs_": "num_epochs",

        "data_": "data",
        "test_data_": "test_data",
        "prompt_": "prompt",

        "group_by_length_": "group_by_length",
        "expand_side_": "expand_side",
    }

    def __init__(self, config: Dict[str, str]):
        super().__init__(config)
        self.init(self.__params_map, config)

        if config["optim"] == "adamw":
            self.optim_config_ = AdamWOptimConfig(config)
        elif config["optim"] == "sgd":
            self.optim_config_ = SGDOptimConfig(config)
        else:
            raise NotImplementedError


class TrainerConfig(DictConfig):
    cutoff_len_: int = 256
    save_step_: int = 2000
    early_stop_test_step_: int = 2000
    train_lora_candidate_num_: int = 2
    train_lora_simultaneously_num_: int = 2
    train_strategy_: str = "optim"

    __params_map: Dict[str, str] = {
        "cutoff_len_": "cutoff_len",
        "save_step_": "save_step",
        "early_stop_test_step_": "early_stop_test_step",
        "train_lora_candidate_num_": "train_lora_candidate_num",
        "train_lora_simultaneously_num_": "train_lora_simultaneously_num",
        "train_strategy_": "train_strategy"
    }

    def __init__(self, config: Dict[str, str]):
        super().__init__(config)
        self.init(self.__params_map, config)


class MLoRAConfig:
    lora_configs_: List[LoraConfig] = None
    trainer_config_: TrainerConfig = None

    def __init__(self, path: str):
        with open(path, 'r', encoding='utf8') as fp:
            config = json.load(fp)

        self.trainer_config_ = TrainerConfig(config)
        self.lora_configs_ = []

        for lora in config["lora"]:
            new_lora_config = LoraConfig(lora)
            new_lora_config.cutoff_len_ = self.trainer_config_.cutoff_len_
            self.lora_configs_.append(new_lora_config)
