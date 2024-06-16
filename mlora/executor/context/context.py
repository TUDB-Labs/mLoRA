from mlora.model.modules import AdapterModel
from mlora.model.args import LinearInfo
from mlora.config import AdapterConfig, OptimizerConfig, LRSchedulerConfig

import torch.optim
from typing import Dict, List, Callable
from collections import OrderedDict
from abc import ABCMeta, abstractmethod


class TaskContext(metaclass=ABCMeta):
    type_: str = ""
    name_: str = ""

    device_: str = ""

    adapter_model_: AdapterModel = {}

    loss_fn_: Callable = None
    optimizer_: torch.optim.Optimizer = None
    lr_scheduler_: torch.optim.lr_scheduler.LRScheduler = None

    def __init__(self, context_type: str, context_name: str) -> None:
        self.type_ = context_type
        self.name_ = context_name

        self.device_ = "cpu"

        self.adapter_model_ = {}

        self.loss_fn_ = None
        self.optimizer_ = None
        self.lr_scheduler_ = None

    @abstractmethod
    def init(self, config: AdapterConfig,
             linears_info: OrderedDict[str, LinearInfo]) -> None:
        ...

    @abstractmethod
    def switch_device(self, device: str) -> None:
        ...

    @abstractmethod
    def step(self) -> None:
        ...

    @abstractmethod
    def weight_dict(self) -> Dict[str, torch.Tensor]:
        ...

    def adapter_model(self) -> AdapterModel:
        return self.adapter_model_

    def set_loss_fn(self, loss_fn: Callable):
        self.loss_fn_ = loss_fn

    def create_optimizer(self, parameters: List[torch.Tensor],
                         optim_config: OptimizerConfig, lr_scheduler_config: LRSchedulerConfig) -> torch.optim:
        if optim_config.optimizer_ == "sgd":
            self.optimizer_ = torch.optim.SGD(
                parameters, lr=float(optim_config.lr_))
        elif optim_config.optimizer_ == "adamw":
            self.optimizer_ = torch.optim.AdamW(
                parameters, lr=float(optim_config.lr_))
        else:
            raise NotImplementedError

        if lr_scheduler_config.lr_scheduler_ == "cosine":
            self.lr_scheduler_ = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_, lr_scheduler_config.t_max_, lr_scheduler_config.eta_min_)
        elif lr_scheduler_config.lr_scheduler_ == "none":
            self.lr_scheduler_ = None
        else:
            raise NotImplementedError

    def switch_tensor(self, data: torch.Tensor, device: str):
        data.data = data.data.to(device)
        if data._grad is None:
            return
        data._grad.data = data._grad.data.to(device)

    def switch_dict_tensor(self, data: Dict, device: str):
        assert isinstance(data, Dict)
        for value in data.values():
            if not isinstance(value, torch.Tensor):
                continue
            self.switch_tensor(value, device)

    def switch_optimizer(self, device: str):
        if self.optimizer_ is None:
            return

        for value in self.optimizer_.state.values():
            if isinstance(value, torch.Tensor):
                self.switch_tensor(value, device)
                continue

            if isinstance(value, Dict):
                self.switch_dict_tensor(value, device)
                continue
