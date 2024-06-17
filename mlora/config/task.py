import logging
from typing import Dict

from .config import DictConfig
from .dataset import DatasetConfig
from .adapter import AdapterConfig


class TaskConfig(DictConfig):
    type_: str = ""

    adapter_: AdapterConfig = None
    dataset_: DatasetConfig = None

    __params_map: Dict[str, str] = {
        "type_": "type",
    }

    def __init__(self, config: Dict[str, str],
                 adapters: Dict[str, AdapterConfig],
                 datasets: Dict[str, DatasetConfig]):
        super().__init__(config)
        self.init(self.__params_map, config)

        self.adapter_ = adapters[config["adapter"]]
        self.dataset_ = datasets[config["dataset"]]


class TrainTaskConfig(TaskConfig):
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
                 adapters: Dict[str, AdapterConfig],
                 datasets: Dict[str, DatasetConfig]):
        super().__init__(config, adapters, datasets)
        self.init(self.__params_map, config)

        assert self.mini_batch_size_ <= self.batch_size_
        assert self.batch_size_ % self.mini_batch_size_ == 0

    @property
    def accumulate_step_(self) -> int:
        return self.batch_size_ // self.mini_batch_size_


class DPOTaskConfig(TrainTaskConfig):
    loss_type_: str = "sigmoid"
    beta_: float = 0.2
    label_smoothing_: float = 0.0

    reference_: AdapterConfig = None

    __params_map: Dict[str, str] = {
        "loss_type_": "loss_type",
        "beta_": "beta",
        "label_smoothing_": "label_smoothing"
    }

    def __init__(self, config: Dict[str, str],
                 adapters: Dict[str, AdapterConfig],
                 datasets: Dict[str, DatasetConfig]):
        super().__init__(config, adapters, datasets)
        self.init(self.__params_map, config)

        if config["reference"] not in adapters:
            self.reference_ = None
            logging.info(
                f"DPOTask - {self.adapter_.name_} use the base model as reference model.")
            return

        self.reference_ = adapters[config["reference"]]


class CPOTaskConfig(TrainTaskConfig):
    loss_type_: str = "sigmoid"
    beta_: float = 0.2

    __params_map: Dict[str, str] = {
        "loss_type_": "loss_type",
        "beta_": "beta"
    }

    def __init__(self, config: Dict[str, str],
                 adapters: Dict[str, AdapterConfig],
                 datasets: Dict[str, DatasetConfig]):
        super().__init__(config, adapters, datasets)
        self.init(self.__params_map, config)


TASKCONFIG_CLASS = {
    "train": TrainTaskConfig,
    "dpo": DPOTaskConfig,
    "cpo": CPOTaskConfig
}
