from typing import Dict

from .config import DictConfig


class DispatcherConfig(DictConfig):
    name_: str = "default"
    concurrency_num_: int = 2

    __params_map: Dict[str, str] = {
        "name_": "name",
        "concurrency_num_": "concurrency_num"
    }

    def __init__(self, config: Dict[str, str]):
        super().__init__(config)
        self.init(self.__params_map, config)
