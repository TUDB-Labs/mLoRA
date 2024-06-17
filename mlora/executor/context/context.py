from mlora.model.modules import AdapterModel
from mlora.model.args import LinearInfo
from mlora.config import AdapterConfig

import torch.optim
from typing import Dict, List, OrderedDict
from abc import ABCMeta, abstractmethod


class TaskContext(metaclass=ABCMeta):
    type_: str = ""
    name_: str = ""
    path_: str = ""

    device_: str = ""

    adapter_model_: AdapterModel = {}

    def __init__(self, context_type: str, context_name: str, context_path: str) -> None:
        self.type_ = context_type
        self.name_ = context_name
        self.path_ = context_path

        self.device_ = "cpu"

        self.adapter_model_ = {}

    @abstractmethod
    def switch_device(self, device: str) -> None:
        ...

    @abstractmethod
    def step(self) -> None:
        ...

    @abstractmethod
    def load_weight(self, config: AdapterConfig, linears_info: OrderedDict[str, LinearInfo]):
        ...

    def adapter_model(self) -> AdapterModel:
        return self.adapter_model_

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

    def switch_list_tensor(self, data: List, device: str):
        assert isinstance(data, List)
        for value in data:
            self.switch_tensor(value, device)
