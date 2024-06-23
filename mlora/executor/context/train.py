from mlora.config import AdapterConfig, OptimizerConfig, LRSchedulerConfig
from mlora.model.args import LinearInfo
import os
import logging
import torch
from abc import abstractmethod
from typing import List, Dict, Callable, Optional
from collections import OrderedDict

from .context import TaskContext

OPTIMIZER_CLASS = {
    "sgd": torch.optim.SGD,
    "adamw": torch.optim.AdamW
}

LR_SCHEDULER_CLASS = {
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
}


class TrainTaskContext(TaskContext):
    loss_fn_: Callable = None
    optimizer_: torch.optim.Optimizer = None
    lr_scheduler_: torch.optim.lr_scheduler.LRScheduler = None

    def __init__(self, config: AdapterConfig, linears_info: OrderedDict[str, LinearInfo]) -> None:
        super().__init__(config.type_, config.name_, config.path_)

        # load the adapter's weight
        self.load_weight(config, linears_info)

        for module in self.adapter_model_.values():
            module.enable_grad()

        # init the optimizer
        self.loss_fn_ = None
        self.optimizer_ = None
        self.lr_scheduler_ = None

        self.create_optimizer(config.optimizer_config_)
        self.create_lr_scheduler(config.lr_scheduler_config_)

    @abstractmethod
    def weight_dict(self) -> Dict[str, torch.Tensor]:
        ...

    def create_optimizer(self, optim_config: OptimizerConfig):
        optimizer_type_ = optim_config.optimizer_
        assert optimizer_type_ in OPTIMIZER_CLASS

        parameters: List[torch.Tensor] = []
        for adapter in self.adapter_model_.values():
            parameters.extend(adapter.get_tensors())

        self.optimizer_ = OPTIMIZER_CLASS[optimizer_type_](
            parameters, **optim_config.to_fn_parameters())
        temp_path = self.path_
        if os.path.isdir(os.path.join(self.path_, "adapters")):
            temp_path = os.path.join(self.path_, "adapters")
            folders = [folder for folder in os.listdir(temp_path)]
            temp_path = os.path.join(temp_path, folders[-1])

        if os.path.isdir(temp_path):
            logging.info(
                f"Adapter {self.name_}:{temp_path} optimizer weight exist, load from file.")
            self.optimizer_.load_state_dict(torch.load(f"{temp_path}{os.sep}optimizer_state.bin"))
        else:
            logging.info(
                f"Adapter {self.name_}:{temp_path} optimizer weight not exist, use the default weight.")
        # load optimizer state
    def create_lr_scheduler(self, lr_scheduler_config: Optional[LRSchedulerConfig]):
        assert self.optimizer_ is not None

        if lr_scheduler_config is None:
            return
        lr_scheduler_config.last_epoch = self.last_epoch
        lr_scheduler_type_ = lr_scheduler_config.lr_scheduler_
        assert lr_scheduler_type_ in LR_SCHEDULER_CLASS
        self.lr_scheduler_ = LR_SCHEDULER_CLASS[lr_scheduler_type_](
            self.optimizer_, **lr_scheduler_config.to_fn_parameters())

    def switch_device(self, device: str) -> None:
        if self.device_ == device:
            return

        for _, adapter in self.adapter_model_.items():
            self.switch_list_tensor(adapter.get_tensors(), device)

        self.switch_optimizer(device)

        self.device_ = device

    def switch_optimizer(self, device: str):
        assert self.optimizer_ is not None

        for value in self.optimizer_.state.values():
            if isinstance(value, torch.Tensor):
                self.switch_tensor(value, device)
                continue

            if isinstance(value, Dict):
                self.switch_dict_tensor(value, device)
                continue

    def step(self) -> None:
        self.optimizer_.step()
        if self.lr_scheduler_ is not None:
            self.lr_scheduler_.step()
        self.optimizer_.zero_grad()

    def set_loss_fn(self, loss_fn: Callable):
        self.loss_fn_ = loss_fn
