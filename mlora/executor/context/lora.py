from mlora.config import LoRAConfig
from mlora.model.modules import LoRA
from mlora.model.args import LinearInfo
import torch
import logging
from collections import OrderedDict
from typing import Dict, override

from .context import TaskContext
from .inference import InferenceTaskContext
from .train import TrainTaskContext


def _load_lora_weight(
    obj: TaskContext,
    config: LoRAConfig,
    linears_info: OrderedDict[str, LinearInfo],
    checkpoint: Dict = None,
):
    # init the weight
    for linear_name, linear_info in linears_info.items():
        target_name = linear_name.split(".")[3]
        if target_name not in config.target_:
            continue
        if config.target_[target_name] is not True:
            continue

        obj.adapter_model_[linear_name] = LoRA(
            config.name_,
            linear_info.in_dim_,
            linear_info.out_dim_,
            config.r_,
            config.alpha_,
            config.dropout_,
        )
    weight_dict = None
    if checkpoint is not None:
        logging.info(f"Adapter {obj.name_}:{obj.path_} weight exist, load from file.")
        weight_dict = checkpoint["lora_weight"]
        prefix_name = "base_model.model.model."
    else:
        logging.info(
            f"Adapter {obj.name_}:{obj.path_} weight not exist, use the default weight."
        )

    for name, module in obj.adapter_model_.items():
        lora_a = (
            None
            if weight_dict is None
            else weight_dict[prefix_name + name + ".lora_A.weight"]
        )
        lora_b = (
            None
            if weight_dict is None
            else weight_dict[prefix_name + name + ".lora_B.weight"]
        )
        module.init_weight(lora_a, lora_b)


class InferenceLoRAContext(InferenceTaskContext):
    config_: LoRAConfig

    def __init__(
        self, config: LoRAConfig, linears_info: OrderedDict[str, LinearInfo]
    ) -> None:
        super().__init__(config, linears_info)

    @override
    def load_weight(self, linears_info: OrderedDict[str, LinearInfo]):
        _load_lora_weight(self, self.config_, linears_info)


class TrainLoRAContext(TrainTaskContext):
    def __init__(
        self,
        config: LoRAConfig,
        linears_info: OrderedDict[str, LinearInfo],
        checkpoint: Dict = None,
    ) -> None:
        super().__init__(config, linears_info, checkpoint)

        self.loss_fn_ = torch.nn.CrossEntropyLoss()

    def load_weight(
        self, linears_info: OrderedDict[str, LinearInfo], checkpoint: Dict = None
    ):
        _load_lora_weight(self, self.config_, linears_info, checkpoint)

    def weight_dict(self) -> Dict[str, torch.Tensor]:
        # base_model.model.model.layers.{0}.self_attn.{q_proj}.{lora_A}.weight
        # base_model.model.model.layers.{0}.mlp.{gate_proj}.{lora_A}.weight
        ret_val = {}
        for name, adapter in self.adapter_model_.items():
            prefix_name = "base_model.model.model." + name
            ret_val[prefix_name + ".lora_A.weight"] = adapter.lora_a_
            ret_val[prefix_name + ".lora_B.weight"] = adapter.lora_b_

        return ret_val

    def checkpoint(self):
        checkpoint = {
            "lora_weight": self.weight_dict(),
            "optimizer": self.optimizer_.state_dict(),
        }
        return checkpoint
