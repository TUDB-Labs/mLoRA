from mlora.modelargs import MultiLoraBatchData
from mlora.LoraLiner import Linear

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class MixGate(torch.nn.Module):
    def __init__(self, adapter_name: str) -> None:
        super().__init__()

        self.adapter_name_: str = adapter_name
        self.gate_: torch.nn.Linear = None
        self.experts_: int = 8
        self.topk_: int = 2

    def set_parameter(self, moe_in_features: int, moe_experts: int, moe_topk: int, device: str):
        self.gate_ = torch.nn.Linear(
            moe_in_features,
            moe_experts,
            bias=False,
            device=device,
        )
        self.experts_ = moe_experts
        self.topk_ = moe_topk

    def routing(self, norm_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # routing to experts
        router_logits = self.gate_.forward(norm_data)
        routing_weights = F.softmax(
            router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.topk_, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        return routing_weights, selected_experts

    def forward(self, routing_state: Tuple[torch.Tensor, torch.Tensor],
                hidden_state: torch.Tensor, expert_idx: int) -> torch.Tensor:
        routing_weights, selected_experts = routing_state
        expert_mask = selected_experts == expert_idx
        expert_weights = (routing_weights * expert_mask).sum(
            dim=-1, keepdim=True
        )
        return hidden_state.mul_(expert_weights)


class MixMoe(torch.nn.Module):
    def __init__(self, w1: Linear, w2: Linear, w3: Linear) -> None:
        super().__init__()

        # feed forward
        self.w1_: Linear = w1  # also gate FNN * dim
        self.w2_: Linear = w2  # also down dim * FNN
        self.w3_: Linear = w3  # also up   FNN * dim
        # mix of experts
        self.moes_: Dict[str, MixGate] = {}

    def init_moe_weight(self, adapter_name: str, moe_in_features: int, moe_experts: int, moe_topk: int, gate: Optional[torch.Tensor] = None):
        if adapter_name not in self.moes_:
            self.moes_[adapter_name] = MixGate(adapter_name)

        self.moes_[adapter_name].set_parameter(
            moe_in_features, moe_experts, moe_topk, self.w1_.device_)

        if gate is not None:
            with torch.no_grad():
                self.moes_[adapter_name].gate_.weight.copy_(gate)

    def forward(self, score_norm_data: torch.Tensor, input_args: MultiLoraBatchData) -> torch.Tensor:
        common_w1 = self.w1_.weight_.forward(score_norm_data)
        common_w3 = self.w3_.weight_.forward(score_norm_data)
        final_ffn_output = None
        for lora_config in input_args.lora_batch_data_config_:
            moe_name = lora_config.adapter_name_
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            if moe_name == "" or moe_name not in self.moes_:
                continue

            moe_layer = self.moes_[moe_name]

            norm_data = score_norm_data[start_idx:end_idx]
            w1_data = common_w1[start_idx:end_idx]
            w3_data = common_w3[start_idx:end_idx]

            routing_state = moe_layer.routing(norm_data)

            # mix of experts
            final_hidden_states = None
            for expert_idx in range(moe_layer.experts_):
                lora_name = f"moe.{moe_name}.experts.{expert_idx}"
                if lora_name in self.w1_.loras_:
                    w1 = w1_data + \
                        self.w1_.loras_[lora_name].forward(norm_data)
                else:
                    w1 = w1_data

                if lora_name in self.w3_.loras_:
                    w3 = w3_data + \
                        self.w3_.loras_[lora_name].forward(norm_data)
                else:
                    w3 = w3_data

                silu_result = F.silu(w1) * w3
                if lora_name in self.w2_.loras_:
                    hidden_state = self.w2_.weight_.forward(
                        silu_result
                    ) + self.w2_.loras_[lora_name].forward(silu_result)
                else:
                    hidden_state = self.w2_.weight_.forward(silu_result)

                current_hidden_states = moe_layer.forward(
                    routing_state, hidden_state, expert_idx)
                if final_hidden_states is None:
                    final_hidden_states = current_hidden_states
                else:
                    final_hidden_states.add_(current_hidden_states)

            if final_ffn_output is None:
                final_ffn_output = final_hidden_states
            else:
                final_ffn_output = torch.cat(
                    [final_ffn_output, final_hidden_states], dim=0
                )

        return final_ffn_output
