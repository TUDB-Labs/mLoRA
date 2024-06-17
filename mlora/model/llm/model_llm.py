from mlora.model.args import LinearInfo, ModelData
from mlora.model.modules import AdapterModel

from collections import OrderedDict
from abc import ABCMeta, abstractmethod


class LLMModel(metaclass=ABCMeta):
    name_or_path_: str = ""
    device_: str = ""
    vocab_size_: int = -1
    n_heads_: int = -1
    dim_: int = -1

    @abstractmethod
    def forward(self, input: ModelData):
        pass

    @abstractmethod
    def load_adapter(self, adapter_model: AdapterModel):
        pass

    @abstractmethod
    def offload_adapter(self, adapter_name: str):
        pass

    @abstractmethod
    def linears_info(self) -> OrderedDict[str, LinearInfo]:
        pass
