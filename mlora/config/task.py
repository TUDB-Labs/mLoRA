import logging
from typing import Dict, Mapping, Optional, Type

from .adapter import AdapterConfig
from .config import DictConfig
from .dataset import DatasetConfig


class TaskConfig(DictConfig):
    name_: str
    type_: str

    __params_map: Dict[str, str] = {
        "name_": "name",
        "type_": "type",
    }

    def __init__(
        self,
        config: Dict[str, str],
        adapters: Mapping[str, AdapterConfig],
        datasets: Mapping[str, DatasetConfig],
    ):
        super().__init__(config)
        self.init(self.__params_map, config)

        self.adapter_ = adapters[config["adapter"]]
        self.dataset_ = datasets[config["dataset"]]


class TrainTaskConfig(TaskConfig):
    batch_size_: int
    mini_batch_size_: int
    num_epochs_: int
    cutoff_len_: int
    save_step_: int

    __params_map: Dict[str, str] = {
        "batch_size_": "batch_size",
        "mini_batch_size_": "mini_batch_size",
        "num_epochs_": "num_epochs",
        "cutoff_len_": "cutoff_len",
        "save_step_": "save_step",
    }

    def __init__(
        self,
        config: Dict[str, str],
        adapters: Mapping[str, AdapterConfig],
        datasets: Mapping[str, DatasetConfig],
    ):
        super().__init__(config, adapters, datasets)
        self.init(self.__params_map, config)

        self.batch_size_ = int(self.batch_size_)
        self.mini_batch_size_ = int(self.mini_batch_size_)
        self.num_epochs_ = int(self.num_epochs_)
        self.cutoff_len_ = int(self.cutoff_len_)
        self.save_step_ = int(self.save_step_)

        assert self.mini_batch_size_ <= self.batch_size_
        assert self.batch_size_ % self.mini_batch_size_ == 0

    @property
    def accumulate_step_(self) -> int:
        return self.batch_size_ // self.mini_batch_size_


class DPOTaskConfig(TrainTaskConfig):
    loss_type_: str
    beta_: float
    label_smoothing_: float

    # is reference is None, use the base model
    reference_: Optional[AdapterConfig]

    __params_map: Dict[str, str] = {
        "loss_type_": "loss_type",
        "beta_": "beta",
        "label_smoothing_": "label_smoothing",
    }

    def __init__(
        self,
        config: Dict[str, str],
        adapters: Mapping[str, AdapterConfig],
        datasets: Mapping[str, DatasetConfig],
    ):
        super().__init__(config, adapters, datasets)
        self.init(self.__params_map, config)

        self.beta_ = float(self.beta_)
        self.label_smoothing_ = float(self.label_smoothing_)

        if config["reference"] not in adapters:
            self.reference_ = None
            logging.info(
                f"DPOTask - {self.adapter_.name_} "
                + "use the base model as reference model."
            )
            return

        self.reference_ = adapters[config["reference"]]


class CPOTaskConfig(TrainTaskConfig):
    loss_type_: str
    beta_: float

    __params_map: Dict[str, str] = {"loss_type_": "loss_type", "beta_": "beta"}

    def __init__(
        self,
        config: Dict[str, str],
        adapters: Mapping[str, AdapterConfig],
        datasets: Mapping[str, DatasetConfig],
    ):
        super().__init__(config, adapters, datasets)
        self.init(self.__params_map, config)

        self.beta_ = float(self.beta_)


TASKCONFIG_CLASS: Dict[str, Type[TaskConfig]] = {
    "train": TrainTaskConfig,
    "dpo": DPOTaskConfig,
    "cpo": CPOTaskConfig,
}
