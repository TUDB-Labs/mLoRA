from typing import Dict

from .config import DictConfig
from .dataset import DatasetConfig
from .adapter import AdapterConfig


class TaskConfig(DictConfig):
    adapter_: AdapterConfig = None
    dataset_: DatasetConfig = None

    batch_size_: int = -1
    mini_batch_size_: int = -1
    num_epochs_: int = -1
    cutoff_len_: int = 256
    save_step_: int = 2000

    __params_map: Dict[str, str] = {
        "batch_size_": "batch_size",
        "mini_batch_size_": "mini_batch_size",
        "num_epochs_": "num_epochs",
        "cutoff_len_": "cutoff_len",
        "save_step_": "save_step"
    }

    def __init__(self, config: Dict[str, str],
                 adapter: AdapterConfig, dataset: DatasetConfig):
        super().__init__(config)
        self.init(self.__params_map, config)

        assert self.mini_batch_size_ % self.batch_size_ == 0

        self.adapter_ = adapter
        self.dataset_ = dataset

    @property
    def accumulate_step(self) -> int:
        return self.batch_size_ // self.mini_batch_size_
