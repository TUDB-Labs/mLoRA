from typing import Dict

from .config import DictConfig
from .optimizer import OptimizerConfig, OPTIMIZERCONFIG_CLASS
from .scheduler import LRSchedulerConfig, LRSCHEDULERCONFIG_CLASS


class AdapterConfig(DictConfig):
    type_: str = ""
    name_: str = ""

    optimizer_config_: OptimizerConfig = None
    lr_scheduler_config_: LRSchedulerConfig = None

    __params_map: Dict[str, str] = {
        "type_": "type",
        "name_": "name",
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

        self.__init_optim(config)
        self.__init_lr_scheduler(config)

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

    def __init__(self, config: Dict[str, any]):
        super().__init__(config)
        self.init(self.__params_map, config)

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


ADAPTERCONFIG_CLASS = {
    "lora": LoRAConfig
}
