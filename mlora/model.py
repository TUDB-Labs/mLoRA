from mlora.modelargs import LoraConfig, MultiLoraBatchData, LLMModelOutput

import torch

from abc import ABCMeta, abstractclassmethod
from typing import Tuple, Dict, List, Optional


class LLMOutput(metaclass=ABCMeta):
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        pass

    def loss(self,
             input_ids: torch.Tensor,
             output_logits: torch.Tensor,
             labels: List[List[int]]) -> torch.Tensor:
        pass

    def state_dict(self):
        return {}


class LLMModel(metaclass=ABCMeta):
    @abstractclassmethod
    def init_lora_layer_weight(self, config: LoraConfig, weight: Optional[Dict[str, torch.Tensor]]):
        pass

    @abstractclassmethod
    def load_adapter_weight(self, path: str, adapter_name: str = None):
        pass

    @abstractclassmethod
    def get_lora_weight_dict(self, lora_name: str) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        pass

    @abstractclassmethod
    def sequential_module(self) -> torch.nn.Sequential:
        pass

    @abstractclassmethod
    def get_generate_paramas(self) -> Dict[str, any]:
        pass

    @abstractclassmethod
    def forward(self, input: MultiLoraBatchData,
                labels: List[List[int]] = None) -> List[LLMModelOutput]:
        pass
