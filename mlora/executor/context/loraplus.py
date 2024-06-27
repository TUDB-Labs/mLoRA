from collections import OrderedDict
from typing import List, override

import torch

from mlora.config import LoRAPlusConfig, OptimizerConfig
from mlora.model.args import LinearInfo

from .lora import TrainLoRAContext
from .train import OPTIMIZER_CLASS


class TrainLoRAPlusContext(TrainLoRAContext):
    lr_ratio_: float

    def __init__(
        self, config: LoRAPlusConfig, linears_info: OrderedDict[str, LinearInfo]
    ) -> None:
        self.lr_ratio_ = float(config.lr_ratio_)

        super().__init__(config, linears_info)

    @override
    def create_optimizer(self, optim_config: OptimizerConfig | None):
        assert optim_config is not None

        optimizer_type_ = optim_config.optimizer_
        assert optimizer_type_ in OPTIMIZER_CLASS

        lora_a_parameters: List[torch.Tensor] = []
        lora_b_parameters: List[torch.Tensor] = []

        for adapter in self.adapter_model_.values():
            lora_a_parameters.append(adapter.lora_a_)
            lora_b_parameters.append(adapter.lora_b_)

        parameters = [
            {"params": lora_a_parameters},
            {
                "params": lora_b_parameters,
                "lr": self.lr_ratio_ * float(optim_config.lr_),
            },
        ]

        self.optimizer_ = OPTIMIZER_CLASS[optimizer_type_](
            parameters, **optim_config.to_fn_parameters()
        )
