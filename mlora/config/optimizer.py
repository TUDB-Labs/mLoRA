from typing import Dict, override
from abc import abstractmethod

from .config import DictConfig


class OptimizerConfig(DictConfig):
    lr_: float = 0.0
    optimizer_: str = ""

    __params_map: Dict[str, str] = {
        "lr_": "lr",
        "optimizer_": "optimizer",
    }

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)

    @abstractmethod
    def to_fn_parameters(self) -> Dict[str, str]:
        ...


class SGDOptimizerConfig(OptimizerConfig):
    momentum_: float = 0.0

    __params_map: Dict[str, str] = {
        "momentum_": "momentum"
    }

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)

    @override
    def to_fn_parameters(self) -> Dict[str, str]:
        return {
            "lr": float(self.lr_),
            "motentum": float(self.momentum_)
        }


class AdamWOptimizerConfig(OptimizerConfig):
    __params_map: Dict[str, str] = {
    }

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)

    @override
    def to_fn_parameters(self) -> Dict[str, str]:
        return {
            "lr": float(self.lr_),
        }


OPTIMIZERCONFIG_CLASS = {
    "sgd": SGDOptimizerConfig,
    "adamw": AdamWOptimizerConfig
}
