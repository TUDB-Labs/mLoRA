from mlora.modelargs import MixConfig, MultiLoraBatchData
from mlora.LoraLiner import Linear
from mlora.MixLoRA import moe_layer_factory
from mlora.model import RMSNorm

import torch
from typing import Tuple, Optional


class FeedForward(torch.nn.Module):
    def __init__(self, norm: RMSNorm, w1: Linear, w2: Linear, w3: Linear, device: str) -> None:
        super().__init__()

        # feed forward
        self.norm_: RMSNorm = norm  # dim
        self.w1_: Linear = w1       # also gate FNN * dim
        self.w2_: Linear = w2       # also down dim * FNN
        self.w3_: Linear = w3       # also up   FNN * dim
        self.act_ = torch.nn.SiLU()
        # device
        self.device_ = device
        # mix of experts
        self.enable_moe_: bool = False
        self.moes_: torch.ModuleDict = {}

    def init_moe_weight(self, in_features: int, config: MixConfig, gate: Optional[torch.Tensor] = None):
        self.moes_[config.adapter_name_] = moe_layer_factory(
            in_features, config)
        if gate is not None:
            with torch.no_grad():
                self.moes_[config.adapter_name_].gate_.weight.copy_(gate)
        else:
            self.moes_[config.adapter_name_].gate_.weight.data.normal_(
                mean=0.0, std=config.initializer_factor_ * 1)
        self.enable_moe_ = True

    def _lora_forward(self, lora_name, act_fn, norm_data):
        # Applying LoRA weights to FFN weights
        if lora_name in self.w1_.loras_:
            w1 = self.w1_.weight_.forward(norm_data) + \
                self.w1_.loras_[lora_name].forward(norm_data)
        else:
            w1 = self.w1_.weight_.forward(norm_data)

        if lora_name in self.w3_.loras_:
            w3 = self.w3_.weight_.forward(norm_data) + \
                self.w3_.loras_[lora_name].forward(norm_data)
        else:
            w3 = self.w3_.weight_.forward(norm_data)

        act_result = act_fn(w1) * w3
        if lora_name in self.w2_.loras_:
            return self.w2_.weight_.forward(act_result) + \
                self.w2_.loras_[lora_name].forward(act_result)
        else:
            return self.w2_.weight_.forward(act_result)

    def _expert_forward(self, moe_name, act_fn, expert_idx, norm_data):
        lora_name = f"moe.{moe_name}.experts.{expert_idx}"
        return self._lora_forward(lora_name, act_fn, norm_data)

    def forward(self, data: torch.Tensor, router_outputs: Tuple, input_args: MultiLoraBatchData) -> torch.Tensor:
        if not self.enable_moe_:
            score_norm_data = self.norm_(data)
            w1 = self.w1_.forward(score_norm_data, input_args)
            w3 = self.w3_.forward(score_norm_data, input_args)
            return self.w2_.forward(self.act_(w1) * w3, input_args)

        final_hidden_states = None
        for idx, lora_config in enumerate(input_args.lora_batch_data_config_):
            moe_name = lora_config.adapter_name_
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            if moe_name in self.moes_:
                current_hidden_states, current_router_outputs = self.moes_[
                    moe_name].forward(self.norm_, self._expert_forward, data[start_idx:end_idx])

                if router_outputs is not None and current_router_outputs is not None:
                    router_outputs[idx].append(current_router_outputs)
            else:
                score_norm_data = self.norm_(data[start_idx:end_idx])
                current_hidden_states = self._lora_forward(
                    moe_name, self.act_, score_norm_data)

            if final_hidden_states is None:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states = torch.cat(
                    [final_hidden_states, current_hidden_states], dim=0)

        return final_hidden_states
