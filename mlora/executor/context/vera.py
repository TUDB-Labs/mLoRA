from collections import OrderedDict
from typing import Dict, Set, override

import torch

from mlora.config import VeRAConfig
from mlora.model.args import LinearInfo
from mlora.model.modules import VeRA, vera_shared_weight

from .context import TaskContext
from .inference import InferenceTaskContext
from .train import TrainTaskContext


def _init_vera_weight(
    context: TaskContext,
    config: VeRAConfig,
    linears_info: OrderedDict[str, LinearInfo],
):
    enable_target: Set = set()
    # init the weight
    for linear_name, linear_info in linears_info.items():
        target_name = linear_name.split(".")[3]
        if target_name not in config.target_:
            continue
        if config.target_[target_name] is not True:
            continue

        enable_target.add(target_name)

        context.adapter_model_[linear_name] = VeRA(
            config.name_,
            target_name,
            linear_info.in_dim_,
            linear_info.out_dim_,
            config.r_,
            config.alpha_,
            config.dropout_,
            config.d_initial_,
        )

    # only init the lora weight once
    for target in enable_target:
        VeRA.init_lora_weight(config.name_, target, None, None)


class InferenceVeRAContext(InferenceTaskContext):
    config_: VeRAConfig

    def __init__(
        self, config: VeRAConfig, linears_info: OrderedDict[str, LinearInfo]
    ) -> None:
        super().__init__(config, linears_info)

    @override
    def load_weight(self, linears_info: OrderedDict[str, LinearInfo]):
        _init_vera_weight(self, self.config_, linears_info)


class TrainVeRAContext(TrainTaskContext):
    config_: VeRAConfig

    def __init__(
        self,
        config: VeRAConfig,
        linears_info: OrderedDict[str, LinearInfo],
    ) -> None:
        super().__init__(config, linears_info)

        self.loss_fn_ = torch.nn.CrossEntropyLoss()

    @override
    def load_weight(self, linears_info: OrderedDict[str, LinearInfo]):
        _init_vera_weight(self, self.config_, linears_info)

    @override
    def weight_dict(self) -> Dict[str, torch.Tensor]:
        # base_model.model.model.layers.{0}.self_attn.{q_proj}.{b_vec}.weight
        # base_model.model.model.layers.{0}.mlp.{gate_proj}.{d_vec}.weight
        ret_val = {}
        prefix_name = "base_model.model.model."
        for name, adapter in self.adapter_model_.items():
            ret_val[prefix_name + name + ".b_vec.weight"] = adapter.b_vec_
            ret_val[prefix_name + name + ".d_vec.weight"] = adapter.d_vec_

        for target_name, enable in self.config_.target_.items():
            if not enable:
                continue
            lora_a, lora_b = vera_shared_weight(self.name_, target_name)
            ret_val[target_name + ".lora_A.weight"] = lora_a
            ret_val[target_name + ".lora_B.weight"] = lora_b

        return ret_val

    @override
    def recover_weight(self, weight_dict: Dict[str, torch.Tensor]):
        assert weight_dict is not None
        prefix_name = "base_model.model.model."
        for name, module in self.adapter_model_.items():
            b_vec = weight_dict[prefix_name + name + ".b_vec.weight"]
            d_vec = weight_dict[prefix_name + name + ".d_vec.weight"]
            module.init_vec_weight(b_vec, d_vec)

        for target_name, enable in self.config_.target_.items():
            if not enable:
                continue
            lora_a = weight_dict[target_name + ".lora_A.weight"]
            lora_b = weight_dict[target_name + ".lora_B.weight"]

            VeRA.init_lora_weight(self.name_, target_name, lora_a, lora_b)
