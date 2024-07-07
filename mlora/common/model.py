from abc import ABCMeta
from collections import OrderedDict
from typing import Dict, List, Optional

import torch

from .modelargs import LLMModelInput, Masks


class LLMAttention(metaclass=ABCMeta):
    @classmethod
    def state_dict(self) -> Dict[str, torch.nn.Module]:
        return {}

    @classmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        pass


class LLMFeedForward(metaclass=ABCMeta):
    @classmethod
    def state_dict(self) -> Dict[str, torch.nn.Module]:
        return {}

    @classmethod
    def _batch_forward(
        self, data: torch.Tensor, input_args: LLMModelInput
    ) -> torch.Tensor:
        pass

    @classmethod
    def _lora_forward(
        self, lora_name: str, act_fn: torch.nn.Module, data: torch.Tensor
    ) -> torch.Tensor:
        pass


class LLMDecoder(metaclass=ABCMeta):
    @classmethod
    def state_dict(self) -> Dict[str, torch.nn.Module]:
        return {}

    @classmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        attention_mask: Optional[torch.Tensor] = None,
        router_logits: Optional[List[List]] = None,
    ):
        pass


class LLMOutput(metaclass=ABCMeta):
    @classmethod
    def state_dict(self) -> Dict[str, torch.nn.Module]:
        return {}

    @classmethod
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        pass

    @classmethod
    def loss(
        self,
        input_ids: torch.Tensor,
        output_logits: torch.Tensor,
        labels: List[List[int]],
    ) -> torch.Tensor:
        pass


class LLMForCausalLM(metaclass=ABCMeta):
    @classmethod
    def decoder_stack(self) -> List[LLMDecoder]:
        pass

    @classmethod
    def sequential_module(self) -> OrderedDict:
        pass

    @classmethod
    def causal_mask(
        self,
        input_tokens: torch.Tensor,
        additional_mask: List[Masks] = None,
        diagonal: int = 1,
    ) -> torch.Tensor:
        pass

    @staticmethod
    def from_pretrained(llm_model, **kwargs):
        pass
