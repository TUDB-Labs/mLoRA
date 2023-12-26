from mlora.modelargs import MultiLoraBatchData
from mlora.LoraLiner import Linear

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def router_z_loss_func(router_logits: torch.Tensor) -> float:
    num_groups, tokens_per_group, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss) / (num_groups * tokens_per_group)


def load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
    num_experts = router_probs.shape[-1]

    # cast the expert indices to int64, otherwise one-hot encoding will fail
    if expert_indices.dtype != torch.int64:
        expert_indices = expert_indices.to(torch.int64)

    if len(expert_indices.shape) == 2:
        expert_indices = expert_indices.unsqueeze(2)

    expert_mask = torch.nn.functional.one_hot(expert_indices, num_experts)

    # For a given token, determine if it was routed to a given expert.
    expert_mask = torch.max(expert_mask, axis=-2).values

    # cast to float32 otherwise mean will fail
    expert_mask = expert_mask.to(torch.float32)
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    router_prob_per_group_and_expert = torch.mean(router_probs, axis=-2)
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert) * (num_experts**2)


def unpack_router_logits(router_outputs):
    total_router_logits = []
    total_expert_indexes = []
    for router_output in router_outputs:
        if len(router_output[0].shape) > 1:
            router_logits, expert_indexes = router_output
            total_router_logits.append(router_logits)
            total_expert_indexes.append(expert_indexes)
    return torch.cat(total_router_logits, dim=1), torch.cat(total_expert_indexes, dim=1)


def switch_router_loss(z_loss_coef, aux_loss_coef, router_outputs):
    router_logits, expert_indexes = unpack_router_logits(router_outputs)
    z_loss = router_z_loss_func(router_logits)
    router_probs = F.softmax(router_logits, dim=-1)
    aux_loss = load_balancing_loss_func(router_probs, expert_indexes)
    return z_loss_coef * z_loss + aux_loss_coef * aux_loss


class BasicMoe(torch.nn.Module):
    def __init__(self, adapter_name: str, moe_in_features: int, moe_experts: int, moe_topk: int, device: str, **kwargs) -> None:
        super().__init__()

        self.adapter_name_: str = adapter_name
        self.gate_ = torch.nn.Linear(
            moe_in_features, moe_experts, bias=False, device=device)
        self.experts_ = moe_experts
        self.topk_ = moe_topk

    def forward(self, expert_fn, norm_data: torch.Tensor) -> torch.Tensor:
        # routing to experts based on softmax and top-k selection
        router_logits = self.gate_(norm_data)
        routing_weights = F.softmax(
            router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.topk_, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # Routing to experts
        final_hidden_states = None
        for expert_idx in range(self.experts_):
            hidden_state = expert_fn(self.adapter_name_, expert_idx, norm_data)
            # do routing by masking the unselected experts
            expert_mask = selected_experts == expert_idx
            expert_weights = (routing_weights * expert_mask).sum(
                dim=-1, keepdim=True
            )
            current_hidden_states = hidden_state.mul_(expert_weights)
            if final_hidden_states is None:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states.add_(current_hidden_states)

        return final_hidden_states, None


class SwitchMoe(torch.nn.Module):
    def __init__(self, adapter_name: str, moe_in_features: int, moe_expert_capacity: int,
                 moe_jitter_noise: float, moe_experts: int, device: str, **kwargs) -> None:
        super().__init__()

        self.adapter_name_: str = adapter_name
        self.gate_ = torch.nn.Linear(
            moe_in_features, moe_experts, bias=False, device=device)
        self.router_z_loss_coef_: float = 0.001
        self.router_aux_loss_coef_: float = 0.001
        # expert_capacity = (tokens_per_batch / num_experts) * capacity_factor
        self.expert_capacity_: int = moe_expert_capacity
        self.jitter_noise_: float = moe_jitter_noise
        self.experts_: int = moe_experts

    def route(self, norm_data: torch.Tensor) -> Tuple:
        if self.jitter_noise_ > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            norm_data = norm_data * torch.empty_like(norm_data).uniform_(
                1.0 - self.jitter_noise_, 1.0 + self.jitter_noise_)

        # Apply Softmax
        router_logits = self.gate_(norm_data)
        router_probs = F.softmax(
            router_logits, dim=1, dtype=torch.float)

        expert_index = torch.argmax(router_probs, dim=-1)
        expert_index = torch.nn.functional.one_hot(
            expert_index, num_classes=self.experts_)

        # Mask tokens outside expert capacity. Sum over each sequence
        token_priority = torch.cumsum(expert_index, dim=-2)
        # mask if the token routed to to the expert will overflow
        expert_capacity_mask = token_priority <= self.expert_capacity_
        expert_index = expert_index * expert_capacity_mask

        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return expert_index, router_probs, router_logits

    def forward(self, expert_fn, norm_data: torch.Tensor) -> Tuple:
        router_mask, router_probs, router_logits = self.route(norm_data)
        expert_index = torch.argmax(router_mask, dim=-1)

        next_states = norm_data.clone()
        for idx in range(self.experts_):
            token_indices = router_mask[:, :, idx].bool()
            next_states[token_indices] = expert_fn(
                self.adapter_name_, idx, norm_data[token_indices]).to(next_states.dtype)

        hidden_states = router_probs * next_states
        return hidden_states, (router_logits, expert_index)


class MLP(torch.nn.Module):
    def __init__(self, w1: Linear, w2: Linear, w3: Linear, device: str) -> None:
        super().__init__()

        # feed forward
        self.w1_: Linear = w1  # also gate FNN * dim
        self.w2_: Linear = w2  # also down dim * FNN
        self.w3_: Linear = w3  # also up   FNN * dim
        # device
        self.device_ = device
        # mix of experts
        self.enable_moe_: bool = False
        self.moes_: torch.ModuleDict = {}

    def init_moe_weight(self, adapter_name: str,
                        moe_layer: torch.nn.Module, gate: Optional[torch.Tensor] = None):
        self.moes_[adapter_name] = moe_layer
        if gate is not None:
            with torch.no_grad():
                self.moes_[adapter_name].gate_.weight.copy_(gate)
        self.enable_moe_ = True

    def _expert_forward(self, moe_name, expert_idx, norm_data):
        lora_name = f"moe.{moe_name}.experts.{expert_idx}"
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

        silu_result = F.silu(w1) * w3
        if lora_name in self.w2_.loras_:
            return self.w2_.weight_.forward(silu_result) + \
                self.w2_.loras_[lora_name].forward(silu_result)
        else:
            return self.w2_.weight_.forward(silu_result)

    def forward(self, score_norm_data: torch.Tensor, router_outputs: Tuple, input_args: MultiLoraBatchData) -> torch.Tensor:
        if not self.enable_moe_:
            w1 = self.w1_.forward(score_norm_data, input_args)
            w3 = self.w3_.forward(score_norm_data, input_args)
            return self.w2_.forward(F.silu(w1) * w3, input_args)

        final_hidden_states = None
        for idx, lora_config in enumerate(input_args.lora_batch_data_config_):
            moe_name = lora_config.adapter_name_
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            if moe_name == "" or moe_name not in self.moes_:
                continue

            current_hidden_states, current_router_outputs = self.moes_[
                moe_name].forward(self._expert_forward, score_norm_data[start_idx:end_idx])

            if current_router_outputs is not None:
                router_outputs[idx].append(current_router_outputs)

            if final_hidden_states is None:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states = torch.cat(
                    [final_hidden_states, current_hidden_states], dim=0)

        return final_hidden_states
