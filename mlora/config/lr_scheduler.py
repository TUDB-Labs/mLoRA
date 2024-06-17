from typing import Dict, override
from abc import abstractmethod

from .config import DictConfig


class LRSchedulerConfig(DictConfig):
    lr_scheduler_: str = ""

    __params_map: Dict[str, str] = {
        "lr_scheduler_": "lrscheduler"
    }

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)

    @abstractmethod
    def to_fn_parameters(self) -> Dict[str, str]:
        ...


class CosineLRSchedulerConfig(LRSchedulerConfig):
    t_max_: int = -1
    eta_min_: int = 0

    __params_map: Dict[str, str] = {
        "t_max_": "t_max",
        "eta_min_": "eta_min"
    }

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)

    @override
    def to_fn_parameters(self) -> Dict[str, str]:
        return {
            "T_max": float(self.t_max_),
            "eta_min": float(self.eta_min_)
        }


LRSCHEDULERCONFIG_CLASS = {
    "cosine": CosineLRSchedulerConfig,
}
