import math
from typing import override

import torch

from .lora import LoRA


class DoRA(LoRA):
    magnitude_: torch.Tensor

    base_weight_: torch.nn.Linear

    def __init__(
        self,
        adapter_name: str,
        in_dim: int,
        out_dim: int,
        r: int,
        alpha: int,
        dropout: float,
        base_weight: torch.nn.Linear,
    ):
        super().__init__(adapter_name, in_dim, out_dim, r, alpha, dropout)
        self.adapter_type_ = "dora"

        # just refer the base weight, do not change it!!!
        self.base_weight_: torch.nn.Linear = base_weight

        self.magnitude_: torch.Tensor = torch.zeros(
            size=(1, out_dim), device="cpu", requires_grad=False, dtype=torch.float32
        )

    @override
    def init_weight(
        self, lora_a: torch.Tensor | None = None, lora_b: torch.Tensor | None = None
    ):
        with torch.no_grad():
            if lora_a is None:
                torch.nn.init.kaiming_normal_(self.lora_a_, a=math.sqrt(5))
            else:
                self.lora_a_.copy_(lora_a)

            if lora_b is not None:
                self.lora_b_.copy_(lora_b)

            self.magnitude_.copy_(self.get_weight_norm())

    def get_weight_norm(self) -> torch.Tensor:
        with torch.no_grad():
            # the dim is out_dim * in_dim
            lora_weight = self.scaling_ * (self.lora_b_ @ self.lora_a_)
            weight = (
                lora_weight.to(self.base_weight_.weight.device)
                + self.base_weight_.weight
            )
            weight = weight.to(self.lora_a_.device)
            weight_norm: torch.Tensor = torch.linalg.norm(weight, dim=1).to(
                weight.dtype
            )

        assert weight_norm.requires_grad is False
        assert weight_norm.grad_fn is None

        return weight_norm

    @override
    def get_all_tensors(self) -> torch.List[torch.Tensor]:
        return [self.lora_a_, self.lora_b_, self.magnitude_]
