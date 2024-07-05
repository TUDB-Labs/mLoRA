from abc import abstractmethod
from typing import List, MutableMapping

import torch


class Adapter(torch.nn.Module):
    adapter_type_: str
    adapter_name_: str

    def __init__(self, adapter_type: str, adapter_name: str):
        super().__init__()

        self.adapter_type_ = adapter_type
        self.adapter_name_ = adapter_name

    @abstractmethod
    def get_trainable_tensors(self) -> List[torch.Tensor]: ...

    @abstractmethod
    def get_all_tensors(self) -> List[torch.Tensor]: ...

    def disable_grad(self):
        for tensor in self.get_trainable_tensors():
            tensor.requires_grad_(False)
            assert tensor.requires_grad is False

    def enable_grad(self):
        for tensor in self.get_trainable_tensors():
            tensor.requires_grad_(True)
            assert tensor.requires_grad is True
            assert tensor.is_leaf


AdapterModel = MutableMapping[str, Adapter]
