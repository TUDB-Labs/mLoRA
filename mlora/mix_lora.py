from mlora.modelargs import MixConfig

import torch
import torch.nn.functional as F

from typing import List, Tuple
from transformers.activations import ACT2FN


def _mixtral_load_balancing_loss_func(gate_logits: List[torch.Tensor], num_experts: int, top_k: int) -> float:
    concatenated_gate_logits = torch.cat(gate_logits, dim=0)

    routing_weights = torch.nn.functional.softmax(
        concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_experts)

    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    overall_loss = torch.sum(
        tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class MixtralRouterLoss(torch.nn.Module):
    def __init__(self, config: MixConfig) -> None:
        super().__init__()
        self.aux_loss_coef = config.router_aux_loss_coef_
        self.experts = config.num_experts_
        self.topk = config.top_k_

    def forward(self, gate_logits) -> torch.Tensor:
        return self.aux_loss_coef * _mixtral_load_balancing_loss_func(gate_logits, self.experts, self.topk)


class MixtralSparseMoe(torch.nn.Module):
    def __init__(self, in_features: int, config: MixConfig) -> None:
        super().__init__()

        self.adapter_name_: str = config.adapter_name
        self.dtype_: torch.dtype = torch.float32
        self.gate_ = torch.nn.Linear(
            in_features, config.num_experts_, bias=False, device=config.device, dtype=self.dtype_)
        self.act_ = ACT2FN[config.act_fn_]
        self.experts_ = config.num_experts_
        self.topk_ = config.top_k_
        self.router_profile_: bool = False
        self.profiler_: List[int] = None

    def _profiling(self,
                   batch_size: int,
                   sequence_length: int,
                   selected_experts: torch.Tensor) -> None:
        if not self.router_profile_:
            return

        router_statistic_ = list(0 for _ in range(self.experts_))
        for selected in selected_experts.tolist():
            for idx in selected:
                router_statistic_[idx] += 1

        if self.profiler_ is None:
            self.profiler_ = list(0 for _ in range(self.experts_))
            for idx in range(self.experts_):
                self.profiler_[idx] = (
                    router_statistic_[idx] / batch_size) / sequence_length
        else:
            for idx in range(self.experts_):
                pressure = (
                    router_statistic_[idx] / batch_size) / sequence_length
                self.profiler_[idx] = (self.profiler_[idx] + pressure) / 2

    def forward(self, expert_fn, hidden_states: torch.Tensor) -> Tuple:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.view(-1, hidden_dim).to(self.dtype_)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate_(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=self.dtype_)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.topk_, dim=-1)

        self._profiling(batch_size, sequence_length, selected_experts)

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=self.dtype_, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.experts_).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.experts_):
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None,
                                          top_x_list].reshape(-1, hidden_dim).to(input_dtype)
            current_hidden_states = expert_fn(
                self.adapter_name_, self.act_, expert_idx, current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(self.dtype_))

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim).to(input_dtype)

        return final_hidden_states, router_logits


def _switch_router_z_loss_func(router_logits: torch.Tensor) -> float:
    num_groups, tokens_per_group, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z**2
    return torch.sum(z_loss) / (num_groups * tokens_per_group)


def _switch_load_balancing_loss_func(router_probs: torch.Tensor, expert_indices: torch.Tensor) -> float:
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


def _switch_unpack_router_logits(router_outputs):
    total_router_logits = []
    total_expert_indexes = []
    for router_output in router_outputs:
        if len(router_output[0].shape) > 1:
            router_logits, expert_indexes = router_output
            total_router_logits.append(router_logits)
            total_expert_indexes.append(expert_indexes)
    return torch.cat(total_router_logits, dim=1), torch.cat(total_expert_indexes, dim=1)


class SwitchRouterLoss(torch.nn.Module):
    def __init__(self, config: MixConfig) -> None:
        super().__init__()
        self.z_loss_coef = config.router_z_loss_coef_
        self.aux_loss_coef = config.router_aux_loss_coef_

    def forward(self, router_outputs) -> torch.Tensor:
        router_logits, expert_indexes = _switch_unpack_router_logits(
            router_outputs)
        z_loss = _switch_router_z_loss_func(router_logits)
        router_probs = F.softmax(router_logits, dim=-1)
        aux_loss = _switch_load_balancing_loss_func(
            router_probs, expert_indexes)
        return self.z_loss_coef * z_loss + self.aux_loss_coef * aux_loss


class SwitchSparseMoe(torch.nn.Module):
    def __init__(self, in_features: int, config: MixConfig) -> None:
        super().__init__()

        self.adapter_name_: str = config.adapter_name
        self.dtype_: torch.dtype = torch.float32
        self.gate_ = torch.nn.Linear(
            in_features, config.num_experts_, bias=False, device=config.device, dtype=self.dtype_)
        self.act_ = ACT2FN[config.act_fn_]
        self.experts_: int = config.num_experts_
        self.dropout_ = torch.nn.Dropout(
            config.ffn_dropout_) if config.ffn_dropout_ > 0 else torch.nn.Identity()
        self.expert_capacity_: int = config.expert_capacity_
        self.jitter_noise_: float = config.jitter_noise_

    def route(self, hidden_states: torch.Tensor) -> Tuple:
        if self.jitter_noise_ > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise_, 1.0 + self.jitter_noise_)

        # Apply Softmax
        router_logits = self.gate_(hidden_states)
        router_probs = F.softmax(
            router_logits, dim=-1, dtype=self.dtype_)

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

    def forward(self, expert_fn, hidden_states: torch.Tensor) -> Tuple:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(self.dtype_)

        router_mask, router_probs, router_logits = self.route(hidden_states)
        expert_index = torch.argmax(router_mask, dim=-1)

        next_states = hidden_states.clone()
        for idx in range(self.experts_):
            token_indices = router_mask[:, :, idx].bool()
            next_states[token_indices] = expert_fn(
                self.adapter_name_, self.act_, idx, hidden_states[token_indices].to(input_dtype)).to(next_states.dtype)

        hidden_states = self.dropout_(
            router_probs * next_states).to(input_dtype)

        return hidden_states, (router_logits, expert_index)


router_loss_dict = {
    "mixtral": MixtralRouterLoss,
    "switch": SwitchRouterLoss
}


def router_loss_factory(config: MixConfig) -> torch.nn.Module:
    if config.routing_strategy_ not in router_loss_dict:
        raise ValueError(
            f"Unknown routing strategy {config.routing_strategy_}")
    return router_loss_dict[config.routing_strategy_](config)


moe_layer_dict = {
    "mixtral": MixtralSparseMoe,
    "switch": SwitchSparseMoe
}


def moe_layer_factory(in_features: int, config: MixConfig) -> torch.nn.Module:
    if config.routing_strategy_ not in router_loss_dict:
        raise ValueError(
            f"Unknown routing strategy {config.routing_strategy_}")
    return moe_layer_dict[config.routing_strategy_](in_features, config)
