from typing import Dict

from .config import DictConfig


class LRSchedulerConfig(DictConfig):
    lr_scheduler_: str = ""

    __params_map: Dict[str, str] = {
        "lr_scheduler_": "lrscheduler"
    }

    def __init__(self, config: Dict[str, str]) -> None:
        super().__init__(config)
        self.init(self.__params_map, config)


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


LRSCHEDULERCONFIG_CLASS = {
    "none": LRSchedulerConfig,
    "cosine": CosineLRSchedulerConfig,
}
