from typing import Dict

from .config import DictConfig


class OptimizerConfig(DictConfig):
    optimizer_: str = ""
    lr_: float = 0.0

    __params_map: Dict[str, str] = {
        "optimizer_": "optimizer",
        "lr_": "lr"
    }

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)


class SGDOptimizerConfig(OptimizerConfig):
    momentum_: float = 0.0

    __params_map: Dict[str, str] = {
        "momentum_": "momentum"
    }

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)


class AdamWOptimizerConfig(OptimizerConfig):
    __params_map: Dict[str, str] = {}

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)


OPTIMIZERCONFIG_CLASS = {
    "sgd": SGDOptimizerConfig,
    "adamw": AdamWOptimizerConfig
}
