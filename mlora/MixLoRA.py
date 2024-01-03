from mlora.modelargs import MixConfig, MultiLoraBatchData
from mlora.LoraLiner import Linear

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional


def _basic_load_balancing_loss_func(gate_logits: List[torch.Tensor], num_experts: int, top_k: int) -> float:
    gate_logits = torch.cat(gate_logits, dim=0)

    routing_weights, selected_experts = torch.topk(
        gate_logits, top_k, dim=-1)
    routing_weights = routing_weights.softmax(dim=-1)

    # cast the expert indices to int64, otherwise one-hot encoding will fail
    if selected_experts.dtype != torch.int64:
        selected_experts = selected_experts.to(torch.int64)

    if len(selected_experts.shape) == 2:
        selected_experts = selected_experts.unsqueeze(2)

    expert_mask = torch.nn.functional.one_hot(
        selected_experts, num_experts)

    # For a given token, determine if it was routed to a given expert.
    expert_mask = torch.max(expert_mask, axis=-2).values

    # cast to float32 otherwise mean will fail
    expert_mask = expert_mask.to(torch.float32)
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    router_prob_per_group_and_expert = torch.mean(routing_weights, axis=-1)
    return torch.mean(tokens_per_group_and_expert * router_prob_per_group_and_expert.unsqueeze(-1)) * (num_experts**2)


class BasicRouterLoss(torch.nn.Module):
    def __init__(self, moe_aux_loss_coef, moe_experts, moe_topk) -> None:
        super().__init__()
        self.aux_loss_coef = moe_aux_loss_coef
        self.experts = moe_experts
        self.topk = moe_topk

    def forward(self, gate_logits) -> torch.Tensor:
        return self.aux_loss_coef * _basic_load_balancing_loss_func(gate_logits, self.experts, self.topk)


class BasicMoe(torch.nn.Module):
    def __init__(self, in_features: int, config: MixConfig) -> None:
        super().__init__()

        self.adapter_name_: str = config.adapter_name_
        self.gate_ = torch.nn.Linear(
            in_features, config.num_experts_, bias=False, device=config.device_)
        self.experts_ = config.num_experts_
        self.topk_ = config.top_k_

    def forward(self, expert_fn, hidden_states: torch.Tensor) -> Tuple:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate_(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.topk_, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
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
                                          top_x_list].reshape(-1, hidden_dim)
            current_routing_weights = routing_weights[top_x_list,
                                                      idx_list, None]
            current_hidden_states = expert_fn(
                self.adapter_name_, expert_idx, current_state)
            current_hidden_states = current_routing_weights * current_hidden_states

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(
                0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim)
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
    def __init__(self, moe_z_loss_coef, moe_aux_loss_coef) -> None:
        super().__init__()
        self.z_loss_coef = moe_z_loss_coef
        self.aux_loss_coef = moe_aux_loss_coef

    def forward(self, router_outputs) -> torch.Tensor:
        router_logits, expert_indexes = _switch_unpack_router_logits(
            router_outputs)
        z_loss = _switch_router_z_loss_func(router_logits)
        router_probs = F.softmax(router_logits, dim=-1)
        aux_loss = _switch_load_balancing_loss_func(
            router_probs, expert_indexes)
        return self.z_loss_coef * z_loss + self.aux_loss_coef * aux_loss


class SwitchMoe(torch.nn.Module):
    def __init__(self,  in_features: int, config: MixConfig) -> None:
        super().__init__()

        self.adapter_name_: str = config.adapter_name_
        self.gate_ = torch.nn.Linear(
            in_features, config.num_experts_, bias=False, device=config.device_)
        self.router_z_loss_coef_: float = 0.001
        self.router_aux_loss_coef_: float = 0.001
        # expert_capacity = (tokens_per_batch / num_experts) * capacity_factor
        self.expert_capacity_: int = config.expert_capacity_
        self.jitter_noise_: float = config.jitter_noise_
        self.experts_: int = config.num_experts_

    def route(self, norm_data: torch.Tensor) -> Tuple:
        input_dtype = norm_data.dtype
        norm_data = norm_data.to(torch.float32)

        if self.jitter_noise_ > 0:
            # Multiply the token inputs by the uniform distribution - adding some noise
            norm_data *= torch.empty_like(norm_data).uniform_(
                1.0 - self.jitter_noise_, 1.0 + self.jitter_noise_)

        # Apply Softmax
        self.gate_ = self.gate_.to(torch.float32)
        router_logits = self.gate_(norm_data)
        router_probs = F.softmax(
            router_logits, dim=-1, dtype=torch.float32).to(input_dtype)

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

    def init_moe_weight(self, in_features: int, config: MixConfig, gate: Optional[torch.Tensor] = None):
        moe_layer_dict = {
            "basic": BasicMoe,
            "switch": SwitchMoe
        }
        self.moes_[config.adapter_name_] = moe_layer_dict[config.routing_strategy_](
            in_features, config)
        if gate is not None:
            with torch.no_grad():
                self.moes_[config.adapter_name_].gate_.weight.copy_(gate)
        self.enable_moe_ = True

    def _lora_forward(self, lora_name, norm_data):
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

    def _expert_forward(self, moe_name, expert_idx, norm_data):
        lora_name = f"moe.{moe_name}.experts.{expert_idx}"
        return self._lora_forward(lora_name, norm_data)

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

            if moe_name in self.moes_:
                current_hidden_states, current_router_outputs = self.moes_[
                    moe_name].forward(self._expert_forward, score_norm_data[start_idx:end_idx])

                if router_outputs is not None and current_router_outputs is not None:
                    router_outputs[idx].append(current_router_outputs)
            else:
                current_hidden_states = self._lora_forward(
                    moe_name, score_norm_data[start_idx:end_idx])

            if final_hidden_states is None:
                final_hidden_states = current_hidden_states
            else:
                final_hidden_states = torch.cat(
                    [final_hidden_states, current_hidden_states], dim=0)

        return final_hidden_states
