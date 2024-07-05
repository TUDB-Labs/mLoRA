import math
from typing import Dict, List, Tuple, override

import torch

from .adapter import Adapter

# vera name, {q_proj: }
SHARED_LORA_A: Dict[str, Dict[str, torch.Tensor]] = {}
SHAERD_LORA_B: Dict[str, Dict[str, torch.Tensor]] = {}


def vera_shared_weight(
    adapter_name: str, target_name: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    global SHARED_LORA_A
    global SHAERD_LORA_B

    return (
        SHARED_LORA_A[adapter_name][target_name],
        SHAERD_LORA_B[adapter_name][target_name],
    )


class VeRA(Adapter):
    b_vec_: torch.Tensor
    d_vec_: torch.Tensor

    r: int
    d_initial_: float

    # the VeRA paper do not use dropout and scaling
    lora_a_: torch.Tensor
    lora_b_: torch.Tensor

    def __init__(
        self,
        adapter_name: str,
        layer_name: str,
        in_dim: int,
        out_dim: int,
        r: int,
        alpha: int,
        dropout: float,
        d_initial: float,
    ):
        super().__init__("vera", adapter_name)

        self.r_ = r
        self.alpha_: int = alpha
        self.dropout_: float = dropout
        self.scaling_: float = alpha / r
        self.d_initial_ = d_initial

        # shared across layers and be frozen
        # only create once
        if adapter_name not in SHARED_LORA_A:
            assert adapter_name not in SHAERD_LORA_B
            SHARED_LORA_A[adapter_name] = {}
            SHAERD_LORA_B[adapter_name] = {}

        if layer_name not in SHARED_LORA_A[adapter_name]:
            assert layer_name not in SHAERD_LORA_B[adapter_name]
            SHARED_LORA_A[adapter_name][layer_name] = torch.zeros(
                size=(r, in_dim), device="cpu", requires_grad=False, dtype=torch.float32
            )
            SHAERD_LORA_B[adapter_name][layer_name] = torch.zeros(
                size=(out_dim, r),
                device="cpu",
                requires_grad=False,
                dtype=torch.float32,
            )

        # just referece the tensor
        self.lora_a_ = SHARED_LORA_A[adapter_name][layer_name]
        self.lora_b_ = SHAERD_LORA_B[adapter_name][layer_name]

        self.b_vec_ = torch.zeros(
            size=(1, out_dim),
            device="cpu",
            requires_grad=False,
            dtype=torch.float32,
        )

        self.d_vec_ = (
            torch.ones(
                size=(1, r), device="cpu", requires_grad=False, dtype=torch.float32
            )
            * self.d_initial_
        )

    def init_vec_weight(
        self,
        b_vec: torch.Tensor | None = None,
        d_vec: torch.Tensor | None = None,
    ):
        with torch.no_grad():
            if b_vec is not None:
                self.b_vec_.copy_(b_vec)
            if d_vec is not None:
                self.d_vec_.copy_(d_vec)

    @staticmethod
    def init_lora_weight(
        adapter_name: str,
        layer_name: str,
        lora_a: torch.Tensor | None = None,
        lora_b: torch.Tensor | None = None,
    ):
        with torch.no_grad():
            weight = SHARED_LORA_A[adapter_name][layer_name]
            if lora_a is None:
                torch.nn.init.kaiming_normal_(weight, a=math.sqrt(5))
            else:
                weight.copy_(lora_a)

            weight = SHAERD_LORA_B[adapter_name][layer_name]
            if lora_b is None:
                torch.nn.init.kaiming_normal_(weight, a=math.sqrt(5))
            else:
                weight.copy_(lora_b)

    @override
    def get_trainable_tensors(self) -> List[torch.Tensor]:
        return [self.b_vec_, self.d_vec_]

    @override
    def get_all_tensors(self) -> List[torch.Tensor]:
        return [
            self.b_vec_,
            self.d_vec_,
            *SHARED_LORA_A[self.adapter_name_].values(),
            *SHAERD_LORA_B[self.adapter_name_].values(),
        ]
