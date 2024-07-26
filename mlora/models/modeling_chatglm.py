import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from transformers.utils import is_flash_attn_2_available

from mlora.backends import backend
from mlora.common import (
    Cache,
    FeedForward,
    Linear,
    LLMAttention,
    LLMDecoder,
    LLMFeedForward,
    LLMForCausalLM,
    LLMModelConfig,
    LLMModelInput,
    flash_attention_forward,
)
from mlora.common.mix_lora import _mixtral_slice_tensor
from mlora.utils import copy_parameters


@dataclass
class GLMConfig(LLMModelConfig):
    post_layer_norm: bool = True
    rmsnorm: bool = True
    layernorm_epsilon: float = 1e-5
    apply_residual_connection_post_layernorm: bool = False
    fp32_residual_connection: bool = False
    kv_channels: int = 128
    multi_query_attention: bool = False
    multi_query_group_num: int = 2
    apply_query_key_layer_scaling: bool = True
    attention_softmax_in_fp32: bool = True
    original_rope: bool = True
    add_bias_linear: bool = False
    padded_vocab_size: int = -1
    rope_ratio: float = 1


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> List[torch.Tensor]:
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, rope_ratio=1, original_impl=False, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl
        self.rope_ratio = rope_ratio

    def forward_impl(
        self,
        seq_len: int,
        n_elem: int,
        dtype: torch.dtype,
        device: torch.device,
        base: int = 10000,
    ):
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
        base = base * self.rope_ratio
        theta = 1.0 / (
            base
            ** (torch.arange(0, n_elem, 2, dtype=torch.float, device=device) / n_elem)
        )

        # Create position indexes `[0, 1, ..., seq_len - 1]`
        seq_idx = torch.arange(seq_len, dtype=torch.float, device=device)

        # Calculate the product of position index and $\theta_i$
        idx_theta = torch.outer(seq_idx, theta).float()

        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

        # this is to mimic the behaviour of complex32, else we will get different results
        if dtype in (torch.float16, torch.bfloat16, torch.int8):
            cache = cache.bfloat16() if dtype == torch.bfloat16 else cache.half()
        return cache

    def forward(self, max_seq_len, offset=0):
        return self.forward_impl(
            max_seq_len,
            self.dim,
            dtype=self.inv_freq.dtype,
            device=self.inv_freq.device,
        )


@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [b, np, sq, hn]
    b, np, sq, _ = x.shape
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:, :sq]
    xshaped = x.reshape(b, np, sq, rot_dim // 2, 2)
    rope_cache = rope_cache.view(-1, 1, sq, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(3)
    return torch.cat((x_out2, x_pass), dim=-1)


class RMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(normalized_shape, device=device, dtype=dtype)
        )
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return (self.weight * hidden_states).to(input_dtype)


class CoreAttention(torch.nn.Module):
    def __init__(self, config: GLMConfig, layer_number):
        super(CoreAttention, self).__init__()
        self.config = config
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.is_causal = True

        projection_size = config.kv_channels * config.n_heads_

        # Per attention head and per partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.n_heads_
        self.num_attention_heads_per_partition = config.n_heads_

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # [b, np, sq, sk]
        output_size = (
            query_layer.size(0),
            query_layer.size(1),
            query_layer.size(2),
            key_layer.size(2),
        )

        # [b, np, sq, hn] -> [b * np, sq, hn]
        query_layer = query_layer.view(
            output_size[0] * output_size[1], output_size[2], -1
        )
        # [b, np, sk, hn] -> [b * np, sk, hn]
        key_layer = key_layer.view(output_size[0] * output_size[1], output_size[3], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=query_layer.device,
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer,  # [b * np, sq, hn]
            key_layer.transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # attention scores and attention mask [b, np, sq, sk]
        if self.attention_softmax_in_fp32:
            attention_scores = attention_scores.float()
        if self.coeff is not None:
            attention_scores = attention_scores * self.coeff
        if (
            attention_mask is None
            and attention_scores.shape[2] == attention_scores.shape[3]
        ):
            attention_mask = torch.ones(
                output_size[0],
                1,
                output_size[2],
                output_size[3],
                device=attention_scores.device,
                dtype=torch.bool,
            )
            attention_mask.tril_()
            attention_mask = ~attention_mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                attention_mask, float("-inf")
            )
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.type_as(value_layer)

        # query layer shape: [b * np, sq, hn]
        # value layer shape: [b, np, sk, hn]
        # attention shape: [b, np, sq, sk]
        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(0),
            value_layer.size(1),
            query_layer.size(1),
            value_layer.size(3),
        )
        # change view [b * np, sk, hn]
        value_layer = value_layer.view(
            output_size[0] * output_size[1], value_layer.size(2), -1
        )
        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )
        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer)
        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        # [b, np, sq, hn] --> [b, sq, np, hn]
        context_layer = context_layer.transpose(1, 2).contiguous()
        # [b, sq, np, hn] --> [b, sq, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.reshape(*new_context_layer_shape)

        return context_layer


class FlashAttention2(CoreAttention):
    def __init__(self, *args, **kwargs):
        assert is_flash_attn_2_available(), "Flash Attention is not available"
        super().__init__(*args, **kwargs)

    def forward(self, query_states, key_states, value_states, attention_mask):
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        batch_size, query_length = query_states.shape[:2]

        attn_output = flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            query_length,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(
            batch_size, query_length, self.hidden_size_per_partition
        ).contiguous()

        return attn_output


CORE_ATTENTION_CLASSES = {
    "eager": CoreAttention,
    "flash_attn": FlashAttention2,
}


class GLMSelfAttention(LLMAttention):
    def __init__(
        self,
        qkv_layer: torch.nn.Module,
        dense_layer: torch.nn.Module,
        config: GLMConfig,
        layer_idx,
    ):
        super(GLMSelfAttention, self).__init__()
        self.layer_idx = layer_idx

        self.projection_size = config.kv_channels * config.n_heads_

        # Per attention head and per-partition values.
        self.hidden_size_per_attention_head = self.projection_size // config.n_heads_
        self.num_attention_heads_per_partition = config.n_heads_
        self.multi_query_attention = config.multi_query_attention
        self.qkv_hidden_size = 3 * self.projection_size

        if self.multi_query_attention:
            self.num_multi_query_groups_per_partition = config.multi_query_group_num
            self.qkv_hidden_size = (
                self.projection_size
                + 2
                * self.hidden_size_per_attention_head
                * self.num_multi_query_groups_per_partition
            )

        # QKV layer.
        self.query_key_value = Linear(base_layer=qkv_layer, device=config.device_)
        # Core attention layer.
        self.core_attention = CORE_ATTENTION_CLASSES[config.attn_implementation_](
            config, self.layer_idx
        )

        # Dense layer.
        self.dense = Linear(base_layer=dense_layer, device=config.device_)

    def state_dict(self) -> Dict[str, Linear]:
        return {"qkv_proj": self.query_key_value, "dense": self.dense}

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        rotary_pos_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
    ):
        mixed_x_layer = self.query_key_value(hidden_states, input_args)

        if self.multi_query_attention:
            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [
                    self.num_attention_heads_per_partition
                    * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition
                    * self.hidden_size_per_attention_head,
                    self.num_multi_query_groups_per_partition
                    * self.hidden_size_per_attention_head,
                ],
                dim=-1,
            )
            query_layer = query_layer.view(
                query_layer.size()[:-1]
                + (
                    self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
            key_layer = key_layer.view(
                key_layer.size()[:-1]
                + (
                    self.num_multi_query_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
            value_layer = value_layer.view(
                value_layer.size()[:-1]
                + (
                    self.num_multi_query_groups_per_partition,
                    self.hidden_size_per_attention_head,
                )
            )
        else:
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [b, sq, np, 3 * hn] --> 3 [b, sq, np, hn]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(
                mixed_x_layer, 3
            )

        # [b, sq, np, hn] -> [b, np, sq, hn]
        query_layer, key_layer, value_layer = [
            k.transpose(1, 2) for k in [query_layer, key_layer, value_layer]
        ]

        # apply relative positional encoding (rotary embedding)
        query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
        key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)

        if past_key_value is not None:
            key_layer, value_layer = past_key_value.update(
                key_layer,
                value_layer,
                self.layer_idx,
                {"cache_position": cache_position},
            )

        if self.multi_query_attention:
            key_layer = key_layer.unsqueeze(2)
            key_layer = key_layer.expand(
                -1,
                -1,
                self.num_attention_heads_per_partition
                // self.num_multi_query_groups_per_partition,
                -1,
                -1,
            )
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:1]
                + (self.num_attention_heads_per_partition,)
                + key_layer.size()[3:]
            )
            value_layer = value_layer.unsqueeze(2)
            value_layer = value_layer.expand(
                -1,
                -1,
                self.num_attention_heads_per_partition
                // self.num_multi_query_groups_per_partition,
                -1,
                -1,
            )
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:1]
                + (self.num_attention_heads_per_partition,)
                + value_layer.size()[3:]
            )

        context_layer = self.core_attention(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
        )

        output = self.dense(context_layer, input_args)

        return output


class GLMMLP(LLMFeedForward):
    def __init__(
        self,
        dense_h_to_4h: torch.nn.Module,
        dense_4h_to_h: torch.nn.Module,
        config: GLMConfig,
    ) -> None:
        super().__init__()
        self.dense_h_to_4h: Linear = Linear(dense_h_to_4h, config.device_)
        self.dense_4h_to_h: Linear = Linear(dense_4h_to_h, config.device_)

        def swiglu(x):
            x = torch.chunk(x, 2, dim=-1)
            return F.silu(x[0]) * x[1]

        self.activation_func = swiglu

    def state_dict(self) -> Dict[str, torch.nn.Module]:
        return {
            "dense_h_to_4h": self.dense_h_to_4h,
            "dense_4h_to_h": self.dense_4h_to_h,
        }

    def _batch_forward(
        self, data: torch.Tensor, input_args: LLMModelInput
    ) -> torch.Tensor:
        # [b, sq, h] -> [b, sq, 4hp]
        intermediate_parallel = self.dense_h_to_4h(data, input_args)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        # [b, sq, 4hp] -> [b, sq, h]
        output = self.dense_4h_to_h(intermediate_parallel, input_args)
        return output

    def _lora_forward(
        self, lora_name: str, act_fn: torch.nn.Module, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if lora_name in self.dense_h_to_4h.loras_:
            hidden_states = self.dense_h_to_4h.loras_[lora_name].forward(
                self.dense_h_to_4h.base_layer_.forward(hidden_states), hidden_states
            )
        else:
            hidden_states = self.dense_h_to_4h.base_layer_.forward(hidden_states)

        hidden_states = self.activation_func(hidden_states)

        if lora_name in self.dense_4h_to_h.loras_:
            hidden_states = self.dense_4h_to_h.loras_[lora_name].forward(
                self.dense_4h_to_h.base_layer_.forward(hidden_states), hidden_states
            )
        else:
            hidden_states = self.dense_4h_to_h.base_layer_.forward(hidden_states)

        return hidden_states

    def _mixlora_forward(
        self, moe_name, act_fn, expert_mask, hidden_states, input_dtype
    ):
        common_dense_h_to_4h = self.dense_h_to_4h.base_layer_.forward(
            hidden_states.to(input_dtype)
        ).to(hidden_states.dtype)
        final_expert_states = []
        for expert_idx in range(expert_mask.shape[0]):
            _, top_x = torch.where(expert_mask[expert_idx])

            lora_name = f"moe.{moe_name}.experts.{expert_idx}"
            if lora_name in self.dense_h_to_4h.loras_:
                lora_data = _mixtral_slice_tensor(hidden_states, top_x, input_dtype)
                act_result = self.activation_func(
                    self.dense_h_to_4h.loras_[lora_name].forward(
                        _mixtral_slice_tensor(common_dense_h_to_4h, top_x, input_dtype),
                        lora_data,
                    )
                )
            else:
                act_result = self.activation_func(
                    _mixtral_slice_tensor(common_dense_h_to_4h, top_x, input_dtype)
                )

            if lora_name in self.dense_4h_to_h.loras_:
                final_expert_states.append(
                    self.dense_4h_to_h.loras_[lora_name].forward(
                        self.dense_4h_to_h.base_layer_.forward(act_result), act_result
                    )
                )
            else:
                final_expert_states.append(
                    self.dense_4h_to_h.base_layer_.forward(act_result)
                )

        return final_expert_states


class GLMDecoderLayer(LLMDecoder):
    def __init__(
        self, self_attention: GLMSelfAttention, mlp: FeedForward, config: GLMConfig
    ) -> None:
        super().__init__()
        self.layer_id_ = self_attention.layer_idx
        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm
        )
        self.fp32_residual_connection = config.fp32_residual_connection

        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        # Input layer norm.
        self.input_layernorm = LayerNormFunc(
            config.dim_,
            eps=config.layernorm_epsilon,
            device=config.device_,
            dtype=config.dtype_,
        )
        # Self-attention layer.
        self.self_attention: GLMSelfAttention = self_attention
        self.hidden_dropout = config.hidden_dropout_

        # Post attention layer norm.
        self.post_layernorm = LayerNormFunc(
            config.dim_,
            eps=config.layernorm_epsilon,
            device=config.device_,
            dtype=config.dtype_,
        )
        # mlp
        self.mlp_: FeedForward = mlp

    def state_dict(self) -> Dict[str, torch.nn.Module]:
        linear_layers = self.self_attention.state_dict()
        linear_layers.update(self.mlp_.state_dict())
        return linear_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        rotary_pos_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
    ):
        layernorm_output = self.input_layernorm(hidden_states)

        attention_output = self.self_attention.forward(
            layernorm_output,
            input_args,
            rotary_pos_emb,
            attention_mask,
            cache_position,
            past_key_value,
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        layernorm_input = F.dropout(
            attention_output,
            p=self.hidden_dropout,
            training=not input_args.inference_mode_,
        )
        layernorm_input = residual + layernorm_input

        # Layer norm post the self attention.
        layernorm_output = self.post_layernorm(layernorm_input)

        # MLP.
        mlp_output, router_logits = self.mlp_(layernorm_output, input_args)

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        output = F.dropout(
            mlp_output, p=self.hidden_dropout, training=not input_args.inference_mode_
        )
        output = residual + output

        return output, *router_logits


class GLMEmbedding(torch.nn.Module):
    def __init__(self, config: GLMConfig):
        super(GLMEmbedding, self).__init__()

        self.hidden_size = config.dim_
        # Word embeddings (parallel).
        self.word_embeddings = nn.Embedding(
            config.padded_vocab_size,
            self.hidden_size,
            dtype=config.dtype_,
            device=config.device_,
        )
        self.fp32_residual_connection = config.fp32_residual_connection

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        embeddings = words_embeddings
        # If the input flag for fp32 residual connection is set, convert for float.
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings


class GLMForCausalLM(LLMForCausalLM):
    def __init__(self, config: GLMConfig) -> None:
        self.config_ = config
        self.padding_idx_ = config.pad_token_id_
        self.vocab_size_ = config.vocab_size_

        # Embedding layer.
        self.embed_tokens_ = GLMEmbedding(config)
        # Rotary Position Embedding.
        self.rotary_emb_layer: RotaryEmbedding = None
        # Encoder(Decoder) layers.
        self.layers_: List[GLMDecoderLayer] = []
        # Final layer norm.
        LayerNormFunc = RMSNorm if config.rmsnorm else LayerNorm
        if self.config_.post_layer_norm:
            self.final_layernorm_ = LayerNormFunc(
                config.dim_,
                eps=config.layernorm_epsilon,
                device=config.device_,
                dtype=config.dtype_,
            )
        else:
            self.final_layernorm_ = nn.Identity()
        # Output layer.
        self.lm_head_ = torch.nn.Linear(
            config.dim_,
            config.vocab_size_,
            bias=config.add_bias_linear,
            dtype=config.dtype_,
            device=config.device_,
        )

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens_(input_ids)

    def rotary_embed(
        self, input_tensor: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rotary_emb_layer(max_seq_len=self.config_.max_seq_len_)[
            None, position_ids[-1]
        ]

    def decoder_stack(self) -> List[LLMDecoder]:
        return self.layers_

    def norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.final_layernorm_(hidden_states)

    def get_masks(
        self,
        input_ids: torch.Tensor,
        past_key_values: Cache,
        padding_mask: torch.Tensor,
    ):
        batch_size, seq_length, _ = input_ids.shape
        full_attention_mask = torch.ones(
            batch_size, seq_length, seq_length, device=input_ids.device
        )
        full_attention_mask.tril_()
        past_length = 0
        if past_key_values:
            past_length = past_key_values.get_seq_length()
        if past_length:
            full_attention_mask = torch.cat(
                (
                    torch.ones(
                        batch_size, seq_length, past_length, device=input_ids.device
                    ),
                    full_attention_mask,
                ),
                dim=-1,
            )
        if padding_mask is not None:
            full_attention_mask = full_attention_mask * padding_mask.unsqueeze(1)
        if not past_length and padding_mask is not None:
            full_attention_mask -= padding_mask.unsqueeze(-1) - 1
        full_attention_mask = (full_attention_mask < 0.5).bool()
        full_attention_mask.unsqueeze_(1)
        return full_attention_mask

    def causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[Cache],
    ) -> torch.Tensor:
        return self.get_masks(input_tensor, past_key_values, attention_mask)

    def model_config(self) -> GLMConfig:
        return self.config_

    @staticmethod
    def from_pretrained(
        llm_model,
        attn_impl: str = "eager",
        use_sliding_window: bool = False,
        device: str = backend.default_device_name(),
    ):
        assert not use_sliding_window, "ChatGLM model does not support SWA."
        # Get the config from LLM model and input args.
        llm_config = llm_model.config
        config = GLMConfig(
            # LLM model args.
            name_or_path_=llm_config._name_or_path,
            device_=device,
            dim_=llm_config.hidden_size,
            head_dim_=llm_config.hidden_size // llm_config.num_attention_heads,
            n_heads_=llm_config.num_attention_heads,
            n_kv_heads_=llm_config.multi_query_group_num,
            n_layers_=llm_config.num_layers,
            hidden_dropout_=llm_config.hidden_dropout,
            vocab_size_=llm_config.vocab_size,
            pad_token_id_=llm_config.pad_token_id,
            max_seq_len_=llm_config.seq_length,
            attn_implementation_=attn_impl,
            dtype_=llm_model.dtype,
            # ChatGLM args.
            post_layer_norm=llm_config.post_layer_norm,
            rmsnorm=llm_config.rmsnorm,
            layernorm_epsilon=llm_config.layernorm_epsilon,
            apply_residual_connection_post_layernorm=llm_config.apply_residual_connection_post_layernorm,
            fp32_residual_connection=llm_config.fp32_residual_connection,
            apply_query_key_layer_scaling=llm_config.apply_query_key_layer_scaling,
            kv_channels=llm_config.kv_channels,
            multi_query_attention=llm_config.multi_query_attention,
            multi_query_group_num=llm_config.multi_query_group_num,
            attention_softmax_in_fp32=llm_config.attention_softmax_in_fp32,
            original_rope=llm_config.original_rope,
            add_bias_linear=llm_config.add_bias_linear,
            padded_vocab_size=llm_config.padded_vocab_size,
            rope_ratio=(
                llm_config.rope_ratio if hasattr(llm_config, "rope_ratio") else 1
            ),
        )

        model = GLMForCausalLM(config)
        llm_model.requires_grad_(False)

        copy_parameters(
            llm_model.transformer.embedding,
            model.embed_tokens_,
        )

        rotary_dim = (
            config.dim_ // config.n_heads_
            if config.kv_channels is None
            else config.kv_channels
        )
        model.rotary_emb_layer = RotaryEmbedding(
            dim=rotary_dim // 2,
            rope_ratio=config.rope_ratio,
            original_impl=config.original_rope,
            device=device,
            dtype=config.dtype_,
        )

        for idx, layer in enumerate(llm_model.transformer.encoder.layers):
            # Get self-attention layer.
            self_attention = GLMSelfAttention(
                qkv_layer=layer.self_attention.query_key_value,
                dense_layer=layer.self_attention.dense,
                config=config,
                layer_idx=idx,
            )
            # Get MLP layer.
            mlp = FeedForward(
                GLMMLP(layer.mlp.dense_h_to_4h, layer.mlp.dense_4h_to_h, config=config)
            )
            # Create a transformer block.
            encoder = GLMDecoderLayer(self_attention, mlp, config)
            copy_parameters(layer.input_layernorm, encoder.input_layernorm)
            copy_parameters(layer.post_attention_layernorm, encoder.post_layernorm)
            model.layers_.append(encoder)

        if config.post_layer_norm:
            copy_parameters(
                llm_model.transformer.encoder.final_layernorm,
                model.final_layernorm_,
            )

        copy_parameters(llm_model.transformer.output_layer, model.lm_head_)

        return model
