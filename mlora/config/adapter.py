import logging
from typing import Dict, Any, override
from abc import abstractmethod

from .config import DictConfig
from .optimizer import OptimizerConfig, OPTIMIZERCONFIG_CLASS
from .lr_scheduler import LRSchedulerConfig, LRSCHEDULERCONFIG_CLASS


class AdapterConfig(DictConfig):
    type_: str = ""
    name_: str = ""
    path_: str = ""

    optimizer_config_: OptimizerConfig = None
    lr_scheduler_config_: LRSchedulerConfig = None

    __params_map: Dict[str, str] = {
        "type_": "type",
        "name_": "name",
        "path_": "path"
    }

    def __init_optim(self, config: Dict[str, str]):
        if config["optimizer"] not in OPTIMIZERCONFIG_CLASS:
            raise NotImplementedError

        self.optimizer_config_ = OPTIMIZERCONFIG_CLASS[config["optimizer"]](
            config)

    def __init_lr_scheduler(self, config: Dict[str, str]):
        if config["lrscheduler"] not in LRSCHEDULERCONFIG_CLASS:
            raise NotImplementedError

        self.lr_scheduler_config_ = LRSCHEDULERCONFIG_CLASS[config["lrscheduler"]](
            config)

    def __init__(self, config: Dict[str, str]):
        super().__init__(config)
        self.init(self.__params_map, config)

        if "optimizer" not in config:
            logging.info(
                f"Adapter {self.name_} without optimizer, only for inference")
            return

        self.__init_optim(config)

        if "lrscheduler" not in config:
            logging.info(
                f"Adapter {self.name_} without lr_scheduler.")
            return

        self.__init_lr_scheduler(config)

    @abstractmethod
    def export(self) -> Dict[str, str]:
        ...


class LoRAConfig(AdapterConfig):
    r_: int = -1
    alpha_: int = 0
    dropout_: float = 0.05
    target_: Dict[str, bool] = {}

    __params_map: Dict[str, str] = {
        "r_": "r",
        "alpha_": "alpha",
        "dropout_": "dropout",
        "target_": "target_modules",
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.init(self.__params_map, config)

    @override
    def export(self) -> Dict[str, str]:
        return {
            "lora_alpha": self.alpha_,
            "lora_dropout": self.dropout_,
            "r": self.r_,
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "bias": "none",
            "target_modules": [key for key in self.target_ if self.target_[key]]
        }


class LoRAPlusConfig(LoRAConfig):
    lr_ratio_: float = 8.0

    __params_map: Dict[str, str] = {
        "lr_ratio_": "lr_ratio"
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.init(self.__params_map, config)


ADAPTERCONFIG_CLASS = {
    "lora": LoRAConfig,
    "loraplus": LoRAPlusConfig
}
