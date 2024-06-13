from mlora.model.args import MLoRABatchData, LinearInfo

from collections import OrderedDict
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlora.trainer.context import TaskContext


class LLMModel(metaclass=ABCMeta):
    name_or_path_: str = ""
    device_: str = ""
    vocab_size_: int = -1
    n_heads_: int = -1
    dim_: int = -1

    @abstractmethod
    def forward(self, input: MLoRABatchData):
        pass

    @abstractmethod
    def load_adapter(self, context: "TaskContext"):
        pass

    @abstractmethod
    def offload_adapter(self, adapter_name: str):
        pass

    @abstractmethod
    def linears_info(self) -> OrderedDict[str, LinearInfo]:
        pass
