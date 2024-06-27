from abc import abstractmethod
from typing import Any, Dict, Type, override

from .config import DictConfig


class OptimizerConfig(DictConfig):
    lr_: float
    optimizer_: str

    __params_map: Dict[str, str] = {
        "lr_": "lr",
        "optimizer_": "optimizer",
    }

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)

        self.lr_ = float(self.lr_)

    @abstractmethod
    def to_fn_parameters(self) -> Dict[str, str]: ...


class SGDOptimizerConfig(OptimizerConfig):
    momentum_: float

    __params_map: Dict[str, str] = {"momentum_": "momentum"}

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)

        self.momentum_ = float(self.momentum_)

    @override
    def to_fn_parameters(self) -> Dict[str, Any]:
        return {"lr": float(self.lr_), "motentum": float(self.momentum_)}


class AdamWOptimizerConfig(OptimizerConfig):
    __params_map: Dict[str, str] = {}

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)

    @override
    def to_fn_parameters(self) -> Dict[str, Any]:
        return {
            "lr": float(self.lr_),
        }


OPTIMIZERCONFIG_CLASS: Dict[str, Type[OptimizerConfig]] = {
    "sgd": SGDOptimizerConfig,
    "adamw": AdamWOptimizerConfig,
}
