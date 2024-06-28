from typing import Dict

from .config import DictConfig


class DispatcherConfig(DictConfig):
    name_: str
    concurrency_num_: int

    __params_map: Dict[str, str] = {
        "name_": "name",
        "concurrency_num_": "concurrency_num",
    }

    def __init__(self, config: Dict[str, str]):
        super().__init__(config)
        self.init(self.__params_map, config)

        self.concurrency_num_ = int(self.concurrency_num_)
