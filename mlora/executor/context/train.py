from abc import abstractmethod
from collections import OrderedDict
from typing import Callable, Dict, List, Type
import Optional
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
        checkpoint: Dict = None,
    ) -> None:
        super().__init__(config)

        # load the adapter's weight
        self.load_weight(linears_info, checkpoint)
        for module in self.adapter_model_.values():
            module.enable_grad()

        self.create_optimizer(config.optimizer_config_)
        self.create_lr_scheduler(config.lr_scheduler_config_)
        if checkpoint is not None:
            self.load_optimizer(checkpoint)

    @abstractmethod
    def weight_dict(self) -> Dict[str, torch.Tensor]: ...

    def create_optimizer(self, optim_config: OptimizerConfig | None):
        assert optim_config is not None

        optimizer_type_ = optim_config.optimizer_
        assert optimizer_type_ in OPTIMIZER_CLASS

        parameters: List[torch.Tensor] = []
        for adapter in self.adapter_model_.values():
            parameters.extend(adapter.get_tensors())

        self.optimizer_ = OPTIMIZER_CLASS[optimizer_type_](
            parameters, **optim_config.to_fn_parameters()
        )

    def load_optimizer(self, checkpoint):
        self.optimizer_.load_state_dict(checkpoint["optimizer"])

    def create_lr_scheduler(
        self,
        lr_scheduler_config: Optional[LRSchedulerConfig] | None,
        checkpoint: Dict = None,
    ):
        assert self.optimizer_ is not None

        if lr_scheduler_config is None:
            self.lr_scheduler_ = None
            return

        lr_scheduler_type_ = lr_scheduler_config.lr_scheduler_
        assert lr_scheduler_type_ in LR_SCHEDULER_CLASS
        if checkpoint is not None:
            self.lr_scheduler_ = LR_SCHEDULER_CLASS[lr_scheduler_type_](
                self.optimizer_,
                **lr_scheduler_config.to_fn_parameters(checkpoint["epoch"]),
            )
        else:
            self.lr_scheduler_ = LR_SCHEDULER_CLASS[lr_scheduler_type_](
                self.optimizer_, **lr_scheduler_config.to_fn_parameters()
            )
            # type: ignore

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
