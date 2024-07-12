from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import List, Optional

import torch

from mlora.model.args import LinearInfo, ModelData
from mlora.model.modules import AdapterModel


class LLMModel(metaclass=ABCMeta):
    name_or_path_: str
    device_: str
    vocab_size_: int
    n_heads_: int
    dim_: int

    @abstractmethod
    def forward(self, input: ModelData): ...

    @staticmethod
    @abstractmethod
    def from_pretrained(
        path: str,
        device: str,
        precision: str,
        partial_model_to_device: Optional[List[int]] = None,
    ) -> "LLMModel": ...

    @abstractmethod
    def load_adapter(self, adapter_model: AdapterModel): ...

    @abstractmethod
    def offload_adapter(self, adapter_name: str): ...

    @abstractmethod
    def linears_info(self) -> OrderedDict[str, LinearInfo]: ...

    @abstractmethod
    def sequential(self) -> torch.nn.Sequential: ...
