from collections import OrderedDict
from typing import override

from mlora.config import DoRAConfig
from mlora.model.args import LinearInfo
from mlora.model.modules import DoRA

from .context import TaskContext
from .lora import InferenceLoRAContext, TrainLoRAContext


def _init_dora_weight(
    context: TaskContext,
    config: DoRAConfig,
    linears_info: OrderedDict[str, LinearInfo],
):
    # init the weight
    for linear_name, linear_info in linears_info.items():
        target_name = linear_name.split(".")[3]
        if target_name not in config.target_:
            continue
        if config.target_[target_name] is not True:
            continue

        context.adapter_model_[linear_name] = DoRA(
            config.name_,
            linear_info.in_dim_,
            linear_info.out_dim_,
            config.r_,
            config.alpha_,
            config.dropout_,
            linear_info.base_weight_,
        )
    for _, module in context.adapter_model_.items():
        module.init_weight(None, None)


class InferenceDoRAContext(InferenceLoRAContext):
    config_: DoRAConfig

    def __init__(
        self, config: DoRAConfig, linears_info: OrderedDict[str, LinearInfo]
    ) -> None:
        super().__init__(config, linears_info)

    @override
    def load_weight(self, linears_info: OrderedDict[str, LinearInfo]):
        _init_dora_weight(self, self.config_, linears_info)


class TrainDoRAContext(TrainLoRAContext):
    config_: DoRAConfig

    def __init__(
        self,
        config: DoRAConfig,
        linears_info: OrderedDict[str, LinearInfo],
    ) -> None:
        super().__init__(config, linears_info)

    @override
    def load_weight(self, linears_info: OrderedDict[str, LinearInfo]):
        _init_dora_weight(self, self.config_, linears_info)
