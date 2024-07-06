import logging
from abc import abstractmethod
from typing import Any, Dict, Optional, override

from .config import DictConfig
from .lr_scheduler import LRSCHEDULERCONFIG_CLASS, LRSchedulerConfig
from .optimizer import OPTIMIZERCONFIG_CLASS, OptimizerConfig


class AdapterConfig(DictConfig):
    type_: str
    name_: str
    path_: str

    optimizer_config_: Optional[OptimizerConfig]
    lr_scheduler_config_: Optional[LRSchedulerConfig]

    __params_map: Dict[str, str] = {"type_": "type", "name_": "name", "path_": "path"}

    def __init_optim(self, config: Dict[str, str]):
        if config["optimizer"] not in OPTIMIZERCONFIG_CLASS:
            raise NotImplementedError

        self.optimizer_config_ = OPTIMIZERCONFIG_CLASS[config["optimizer"]](config)

    def __init_lr_scheduler(self, config: Dict[str, str]):
        if config["lrscheduler"] not in LRSCHEDULERCONFIG_CLASS:
            raise NotImplementedError

        self.lr_scheduler_config_ = LRSCHEDULERCONFIG_CLASS[config["lrscheduler"]](
            config
        )

    def __init__(self, config: Dict[str, str]):
        super().__init__(config)
        self.init(self.__params_map, config)

        self.lr_scheduler_config_ = None
        self.optimizer_config_ = None

        if "optimizer" not in config:
            logging.info(f"Adapter {self.name_} without optimizer, only for inference")
            return

        self.__init_optim(config)

        if "lrscheduler" not in config:
            logging.info(f"Adapter {self.name_} without lr_scheduler.")
            return

        self.__init_lr_scheduler(config)

    @abstractmethod
    def export(self) -> Dict[str, str]: ...


class LoRAConfig(AdapterConfig):
    r_: int
    alpha_: int
    dropout_: float
    target_: Dict[str, bool]

    __params_map: Dict[str, str] = {
        "r_": "r",
        "alpha_": "alpha",
        "dropout_": "dropout",
        "target_": "target_modules",
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.init(self.__params_map, config)

        self.r_ = int(self.r_)
        self.alpha_ = int(self.alpha_)
        self.dropout_ = float(self.dropout_)

        for key, value in self.target_.items():
            self.target_[key] = bool(value)

    @override
    def export(self) -> Dict[str, Any]:
        return {
            "lora_alpha": self.alpha_,
            "lora_dropout": self.dropout_,
            "r": self.r_,
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "bias": "none",
            "target_modules": [key for key in self.target_ if self.target_[key]],
        }


class LoRAPlusConfig(LoRAConfig):
    lr_ratio_: float

    __params_map: Dict[str, str] = {"lr_ratio_": "lr_ratio"}

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.init(self.__params_map, config)

        self.lr_ratio_ = float(self.lr_ratio_)


class VeRAConfig(LoRAConfig):
    d_initial_: float

    __params_map: Dict[str, str] = {
        "d_initial_": "d_initial",
    }

    def __init__(self, config: Dict[str, str]):
        super().__init__(config)
        self.init(self.__params_map, config)

        self.d_initial_ = float(self.d_initial_)

    @override
    def export(self) -> Dict[str, Any]:
        return {
            "lora_alpha": self.alpha_,
            "lora_dropout": self.dropout_,
            "r": self.r_,
            "d_initial": self.d_initial_,
            "peft_type": "VeRA",
            "task_type": "CAUSAL_LM",
            "bias": "none",
            "target_modules": [key for key in self.target_ if self.target_[key]],
        }


class DoRAConfig(LoRAConfig):
    __params_map: Dict[str, str] = {}

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.init(self.__params_map, config)


ADAPTERCONFIG_CLASS = {
    "lora": LoRAConfig,
    "loraplus": LoRAPlusConfig,
    "vera": VeRAConfig,
    "dora": DoRAConfig,
}
