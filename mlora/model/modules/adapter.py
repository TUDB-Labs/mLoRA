import torch

from typing import Dict


class Adapter(torch.nn.Module):
    adapter_type_: str = ""
    adapter_name_: str = ""

    def __init__(self, adapter_type: str, adapter_name: str):
        super().__init__()

        self.adapter_type_ = adapter_type
        self.adapter_name_ = adapter_name


AdapterModel = Dict[str, Adapter]
