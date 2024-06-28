from abc import abstractmethod
from typing import Dict, override

from .config import DictConfig


class LRSchedulerConfig(DictConfig):
    lr_scheduler_: str

    __params_map: Dict[str, str] = {"lr_scheduler_": "lrscheduler"}

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)

    @abstractmethod
    def to_fn_parameters(self) -> Dict[str, str]: ...


class CosineLRSchedulerConfig(LRSchedulerConfig):
    t_max_: int
    eta_min_: int

    __params_map: Dict[str, str] = {"t_max_": "t_max", "eta_min_": "eta_min"}

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)

        self.t_max_ = int(self.t_max_)
        self.eta_min_ = int(self.eta_min_)

    @override
    def to_fn_parameters(self, now_epoch: int = None) -> Dict[str, str]:
        if now_epoch is None:
            return {"T_max": float(self.t_max_), "eta_min": float(self.eta_min_)}
        else:
            return {
                "T_max": float(self.t_max_),
                "eta_min": float(self.eta_min_),
                "last_epoch": int(self.last_epoch),
            }


LRSCHEDULERCONFIG_CLASS = {
    "cosine": CosineLRSchedulerConfig,
}
