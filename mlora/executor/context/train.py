from abc import abstractmethod
from collections import OrderedDict
from typing import Callable, Dict, List, Type

import torch

from mlora.config import AdapterConfig, LRSchedulerConfig, OptimizerConfig
from mlora.model.args import LinearInfo

from .context import TaskContext

OPTIMIZER_CLASS = {"sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}

LR_SCHEDULER_CLASS: Dict[str, Type[torch.optim.lr_scheduler.LRScheduler]] = {
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
}


class TrainTaskContext(TaskContext):
    loss_fn_: Callable
    optimizer_: torch.optim.Optimizer
    lr_scheduler_: torch.optim.lr_scheduler.LRScheduler | None

    def __init__(
        self,
        config: AdapterConfig,
        linears_info: OrderedDict[str, LinearInfo],
    ) -> None:
        super().__init__(config)

        # load the adapter's weight
        self.load_weight(linears_info)
        for module in self.adapter_model_.values():
            module.enable_grad()

        self.create_optimizer(config.optimizer_config_)
        self.create_lr_scheduler(config.lr_scheduler_config_)

    @abstractmethod
    def weight_dict(self) -> Dict[str, torch.Tensor]: ...

    @abstractmethod
    def recover_weight(self, weight_dict: Dict[str, torch.Tensor]): ...

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.optimizer_.state_dict()

    def recover_optimizer(self, state_dict: Dict[str, torch.Tensor]):
        assert self.optimizer_ is not None
        self.optimizer_.load_state_dict(state_dict)

    def recover_lr(self, last_epoch: int):
        # the last_epoch is increased every time you call .step() of scheduler
        # different from the train epoch, be careful
        if self.lr_scheduler_ is None:
            return

        # we recreate the lr scheduler
        self.create_lr_scheduler(self.config_.lr_scheduler_config_, last_epoch)

    def create_optimizer(self, optim_config: OptimizerConfig | None):
        assert optim_config is not None

        optimizer_type_ = optim_config.optimizer_
        assert optimizer_type_ in OPTIMIZER_CLASS

        parameters: List[torch.Tensor] = []
        for adapter in self.adapter_model_.values():
            parameters.extend(adapter.get_trainable_tensors())

        self.optimizer_ = OPTIMIZER_CLASS[optimizer_type_](
            parameters, **optim_config.to_fn_parameters()
        )

    def create_lr_scheduler(
        self, lr_scheduler_config: LRSchedulerConfig | None, last_epoch: int = -1
    ):
        assert self.optimizer_ is not None

        if lr_scheduler_config is None:
            self.lr_scheduler_ = None
            return

        lr_scheduler_type_ = lr_scheduler_config.lr_scheduler_

        kwargs = lr_scheduler_config.to_fn_parameters()
        kwargs["last_epoch"] = last_epoch

        self.lr_scheduler_ = LR_SCHEDULER_CLASS[lr_scheduler_type_](
            self.optimizer_,
            **kwargs,  # type: ignore
        )

    def switch_device(self, device: str) -> None:
        if self.device_ == device:
            return

        for _, adapter in self.adapter_model_.items():
            self.switch_list_tensor(adapter.get_all_tensors(), device)

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
