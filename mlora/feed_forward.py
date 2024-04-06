from mlora.modelargs import LLMModelArgs, MixConfig, MultiLoraBatchData
from mlora.lora_linear import Linear
from mlora.mix_lora import moe_layer_factory

from typing import List, Optional
import torch


class FeedForward(torch.nn.Module):
    def __init__(self, w1: Linear, w2: Linear, w3: Linear, args: LLMModelArgs) -> None:
        super().__init__()

        # feed forward
        self.w1_: Linear = w1       # also gate FNN * dim
        self.w2_: Linear = w2       # also down dim * FNN
        self.w3_: Linear = w3       # also up   FNN * dim
        self.act_ = torch.nn.SiLU()
        # device
        self.device_ = args.device_
        # mix of experts
        self.moes_: torch.ModuleDict = {}

    def forward(self, data: torch.Tensor, input_args: MultiLoraBatchData,
                router_logits: List[List] = None) -> torch.Tensor:
        if len(self.moes_) == 0:
            w1 = self.w1_.forward(data, input_args)
            w3 = self.w3_.forward(data, input_args)
            return self.w2_.forward(self.act_(w1) * w3, input_args)
        else:
            return self._mixlora_forward(data, input_args, router_logits)

    # LoRA
    def _lora_forward(self, lora_name, act_fn, data):
        # Applying LoRA weights to FFN weights
        if lora_name in self.w1_.loras_:
            w1 = self.w1_.loras_[lora_name].forward(
                self.w1_.base_layer_.forward(data), data)
        else:
            w1 = self.w1_.base_layer_.forward(data)

        if lora_name in self.w3_.loras_:
            w3 = self.w3_.loras_[lora_name].forward(
                self.w3_.base_layer_.forward(data), data)
        else:
            w3 = self.w3_.base_layer_.forward(data)

        act_result = act_fn(w1) * w3
        if lora_name in self.w2_.loras_:
            return self.w2_.loras_[lora_name].forward(
                self.w2_.base_layer_.forward(act_result), act_result)
        else:
            return self.w2_.base_layer_.forward(act_result)

    # MixLoRA
    def init_moe_weight(self, in_features: int, config: MixConfig, gate: Optional[torch.Tensor] = None):
        self.moes_[config.adapter_name] = moe_layer_factory(
            in_features, config)
        if gate is None:
            torch.nn.init.normal_(
                self.moes_[config.adapter_name].gate_.weight, mean=0.0, std=config.router_init_range_)
        else:
            with torch.no_grad():
                self.moes_[config.adapter_name].gate_.weight.copy_(gate)

    def _expert_forward_callback(self, moe_name, act_fn, expert_idx, data):
        lora_name = f"moe.{moe_name}.experts.{expert_idx}"
        return self._lora_forward(lora_name, act_fn, data)

    def _mixlora_forward(self, data: torch.Tensor, input_args: MultiLoraBatchData,
                         router_logits: List[List] = None):
        final_hidden_states = None
        for idx, lora_config in enumerate(input_args.lora_batch_data_config_):
            moe_name = lora_config.adapter_name_
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            if moe_name in self.moes_:
                current_hidden_states, current_router_outputs = self.moes_[
                    moe_name].forward(self._expert_forward_callback, data[start_idx:end_idx])

                if router_logits is not None and current_router_outputs is not None:
                    router_logits[idx].append(current_router_outputs)
            else:
                current_hidden_states = self._lora_forward(
                    moe_name, self.act_, data[start_idx:end_idx])

            if final_hidden_states is None:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states = torch.cat(
                    [final_hidden_states, current_hidden_states], dim=0)

        return final_hidden_states
