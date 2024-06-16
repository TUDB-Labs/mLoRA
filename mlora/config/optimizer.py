from typing import Dict

from .config import DictConfig


class OptimizerConfig(DictConfig):
    optimizer_: str = ""

    __params_map: Dict[str, str] = {
        "optimizer_": "optimizer",
    }

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)


class SGDOptimizerConfig(OptimizerConfig):
    lr_: float = 0.0
    momentum_: float = 0.0

    __params_map: Dict[str, str] = {
        "lr_": "lr",
        "momentum_": "momentum"
    }

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)


class AdamWOptimizerConfig(OptimizerConfig):
    lr_: float = 0.0

    __params_map: Dict[str, str] = {
        "lr_": "lr"
    }

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)


OPTIMIZERCONFIG_CLASS = {
    "none": OptimizerConfig,
    "sgd": SGDOptimizerConfig,
    "adamw": AdamWOptimizerConfig
}
