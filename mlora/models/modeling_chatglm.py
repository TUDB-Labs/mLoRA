import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import is_flash_attn_2_available

from mlora.backends import get_backend
from mlora.common import (
    CHECKPOINT_CLASSES,
    FeedForward,
    Linear,
    LLMAttention,
    LLMDecoder,
    LLMFeedForward,
    LLMForCausalLM,
    LLMModelArgs,
    LLMModelInput,
    Masks,
    get_unpad_data,
    prepare_4d_causal_attention_mask,
)
from mlora.utils import copy_parameters

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input


@dataclass
class GLMConfig(LLMModelArgs):
    post_layer_norm: bool = True
    rmsnorm: bool = True
    layernorm_epsilon: float = 1e-5
    apply_residual_connection_post_layernorm: bool = False
    attention_dropout: float = 0.0
    fp32_residual_connection: bool = False
    kv_channels: int = 128
    multi_query_attention: bool = False
    multi_query_group_num: int = 2
    apply_query_key_layer_scaling: bool = True
    attention_softmax_in_fp32: bool = True
    original_rope: bool = True
    add_bias_linear: bool = False
    padded_vocab_size: int = -1


def split_tensor_along_last_dim(
    tensor: torch.Tensor,
    num_partitions: int,
    contiguous_split_chunks: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Split a tensor along its last dimension.

    Arguments:
        tensor: input tensor.
        num_partitions: number of partitions to split the tensor
        contiguous_split_chunks: If True, make each chunk contiguous
                                 in memory.

    Returns:
        A list of Tensors
    """
    # Get the size and dimension.
    last_dim = tensor.dim() - 1
    last_dim_size = tensor.size()[last_dim] // num_partitions
    # Split.
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, original_impl=False, device=None, dtype=None):
        super().__init__()
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, dim, 2, device=device).to(dtype=dtype) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.dim = dim
        self.original_impl = original_impl

    def forward_impl(
        self,
        seq_len: int,
        n_elem: int,
        dtype: torch.dtype,
        device: torch.device,
        base: int = 10000,
    ):
        """Enhanced Transformer with Rotary Position Embedding.

        Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
        transformers/rope/__init__.py. MIT License:
        https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
        """
        # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
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
    b, np, sq = x.size(0), x.size(1), x.size(2)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
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
    def __init__(self, dim_, eps, device, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(dim_, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        input_dtype = data.dtype
        variance = data.to(torch.float32).pow(2).mean(-1, keepdim=True)
        data = data * torch.rsqrt(variance + self.eps)
        return (self.weight * data).to(input_dtype)


class GLMLayerNorm(torch.nn.Module):
    def __init__(self, config: GLMConfig):
        super().__init__()
        if config.rmsnorm:
            self.layernorm = RMSNorm(
                dim_=config.dim_,
                eps=config.layernorm_epsilon,
                device=config.device_,
                dtype=config.dtype_,
            )
        else:
            self.layernorm = nn.LayerNorm(
                normalized_shape=config.dim_, eps=config.layernorm_epsilon
            )

    def forward(self, data: torch.tensor) -> torch.Tensor:
        return self.layernorm.forward(data)

    def load_weight(self, weight: torch.nn.Parameter):
        self.layernorm.weight = torch.nn.Parameter(torch.clone(weight))
        self.layernorm.requires_grad_(False)


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

        # Per-attention head and per-partition values.
        self.hidden_size_per_partition = projection_size
        self.hidden_size_per_attention_head = projection_size // config.n_heads_
        self.num_attention_heads_per_partition = config.n_heads_

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff
        self.coeff = coeff

        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def forward(self, query_layer, key_layer, value_layer, attention_mask):
        # [b, np, sq, sk]
        output_size = (
            query_layer.size(0),
            query_layer.size(1),
            query_layer.size(2),
            key_layer.size(2),
        )

        # query_layer: [b, np, sq, hn] -> [b * np, sq, hn]
        query_layer = query_layer.view(
            output_size[0] * output_size[1], output_size[2], -1
        )
        # key_layer: [b, np, sk, hn] -> [b * np, sk, hn]
        key_layer = key_layer.view(output_size[0] * output_size[1], output_size[3], -1)

        # pre-allocating input tensor: [b * np, sq, sk]
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

        # ===========================
        # Attention probs and dropout
        # ===========================

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
                attention_mask.bool(), float("-inf")
            )
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = attention_probs.type_as(value_layer)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

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


class CoreFlashAttention2(CoreAttention):
    def __init__(self, *args, **kwargs):
        assert is_flash_attn_2_available(), "Flash Attention is not available."
        super().__init__(*args, **kwargs)

    def forward(self, query_layer, key_layer, value_layer, attention_mask, dropout=0.0):
        query_states = query_layer.transpose(1, 2)
        key_states = key_layer.transpose(1, 2)
        value_states = value_layer.transpose(1, 2)
        batch_size, query_length = query_states.shape[:2]

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._unpad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=None,
                causal=self.is_causal,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            attn_output = flash_attn_func(
                query_states,
                key_states,
                value_states,
                dropout,
                softmax_scale=None,
                causal=self.is_causal,
            )

        attn_output = attn_output.reshape(
            batch_size, query_length, self.hidden_size_per_partition
        ).contiguous()
        return attn_output

    def _unpad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(
                    batch_size * kv_seq_len,
                    self.num_attention_heads_per_partition,
                    head_dim,
                ),
                indices_k,
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            (
                query_layer,
                indices_q,
                cu_seqlens_q,
                max_seqlen_in_batch_q,
            ) = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


CORE_ATTENTION_CLASSES = {
    "eager": CoreAttention,
    "flash_attn": CoreFlashAttention2,
}


class GLMSelfAttention(LLMAttention):
    def __init__(
        self,
        qkv_layer: torch.nn.Module,
        dense_layer: torch.nn.Module,
        rotary_pos_emb: torch.Tensor,
        config: GLMConfig,
        layer_idx,
    ):
        super(GLMSelfAttention, self).__init__()
        self.layer_idx = max(1, layer_idx)

        self.projection_size = config.kv_channels * config.n_heads_

        # Per attention head and per-partition values.
        self.hidden_size_per_attention_head = (
            config.kv_channels * config.n_heads_ // config.n_heads_
        )
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

        # The rotary position embedding is going to be used for later self-attention mechanism.
        self.rotary_pos_emb = rotary_pos_emb

    def state_dict(self) -> Dict[str, Linear]:
        return {"qkv_proj": self.query_key_value, "dense": self.dense}

    def _split_qkv_tensor(self, mixed_x_layer) -> Tuple[torch.Tensor, ...]:
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
            # [batch, sequence, heads_per_part, 3 * hidden_size_per_head]]
            new_tensor_shape = mixed_x_layer.size()[:-1] + (
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [batch, sequence, heads_per_part, 3 * hidden_size_per_head] ->
            # 3 * [batch, sequence, heads_per_part, hidden_size_per_head]
            (query_layer, key_layer, value_layer) = split_tensor_along_last_dim(
                mixed_x_layer, 3
            )
        return query_layer, key_layer, value_layer

    def _repeat_kv(self, layer, n_rep):
        layer = layer.unsqueeze(2)
        layer = layer.expand(-1, -1, n_rep, -1, -1)
        layer = layer.contiguous().view(
            layer.size()[:1]
            + (self.num_attention_heads_per_partition,)
            + layer.size()[3:]
        )
        return layer

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        # hidden_states: [batch, sequence, hidden_size]

        # =============================================
        #                   QKV Layer
        # =============================================
        # Attention heads [b, sq, h] --> [b, sq, (3 * np * hn)]
        mixed_x_layer = self.query_key_value(hidden_states, input_args)
        # Split the tensor into query, key, and value tensors.
        (query_layer, key_layer, value_layer) = self._split_qkv_tensor(mixed_x_layer)

        # Swap positions of `sequence` and `num_partitions`.
        # [b, sq, np, hn] -> [b, np, sq, hn]
        query_layer, key_layer, value_layer = (
            layer.transpose(1, 2) for layer in [query_layer, key_layer, value_layer]
        )

        # Apply relative positional encoding (rotary embedding).
        if self.rotary_pos_emb is not None:
            query_layer = apply_rotary_pos_emb(query_layer, self.rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, self.rotary_pos_emb)

        if self.multi_query_attention:
            # Expand the kv(group * hidden_size) -> (n_head * hidden_size).
            # kv: [b, s, group_num, hidden_size]-> [b, s, heads, hidden_size]
            n_rep = (
                self.num_attention_heads_per_partition
                // self.num_multi_query_groups_per_partition
            )
            key_layer = self._repeat_kv(key_layer, n_rep)
            value_layer = self._repeat_kv(value_layer, n_rep)

        # =============================================
        #               Core Attention Layer
        # =============================================
        context_layer = self.core_attention(
            query_layer, key_layer, value_layer, attention_mask
        )

        # =============================================
        #                   Dense Layer
        # =============================================
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


class GLMBlock(LLMDecoder):
    def __init__(
        self, self_attention: GLMSelfAttention, mlp: FeedForward, config: GLMConfig
    ) -> None:
        super().__init__()
        self.layer_id_ = self_attention.layer_idx
        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm
        )
        self.fp32_residual_connection = config.fp32_residual_connection
        self.hidden_dropout = config.hidden_dropout_

        # Input layer norm.
        self.input_layernorm = GLMLayerNorm(config)
        # Self-attention layer.
        self.self_attention: GLMSelfAttention = self_attention
        # Post attention layer norm.
        self.post_layernorm = GLMLayerNorm(config)
        # mlp
        self.mlp_: FeedForward = mlp

    def state_dict(self) -> Dict[str, torch.nn.Module]:
        linear_layers = self.self_attention.state_dict()
        linear_layers.update(self.mlp_.state_dict())
        return linear_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        input_args: LLMModelInput,
    ):
        # hidden_states: [b, s, h]

        # =============================================
        #               Input Layer Norm
        # =============================================
        # Layer norm at the beginning of the transformer layer.
        input_norm_result = self.input_layernorm(hidden_states)

        # =============================================
        #            Self-Attention Layer
        # =============================================
        attention_output = self.self_attention.forward(
            hidden_states=input_norm_result,
            input_args=input_args,
            attention_mask=attention_mask,
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = input_norm_result
        else:
            residual = hidden_states

        dropout_result = F.dropout(
            attention_output,
            p=self.hidden_dropout,
            training=not input_args.inference_mode_,
        )
        layernorm_input = residual + dropout_result

        # =============================================
        #           Post Attention Layer Norm
        # =============================================
        layernorm_output = self.post_layernorm(layernorm_input)

        # =============================================
        #                   MLP Layer
        # =============================================
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
        super().__init__()
        self.hidden_size = config.dim_
        self.fp32_residual_connection = config.fp32_residual_connection

        # Embedding layer.
        self.embed_tokens = torch.nn.Embedding(
            config.padded_vocab_size,
            self.hidden_size,
            config.pad_token_id_,
            dtype=config.dtype_,
            device=config.device_,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # ==================================
        #           Embedding Layer
        # ==================================
        embeddings = self.embed_tokens(input_ids)
        if self.fp32_residual_connection:
            embeddings = embeddings.float()
        return embeddings


class GLMSequentialWrapper(torch.nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.wrapper_module_ = module

    def name(self) -> str:
        return type(self.wrapper_module_).__name__

    def forward(self, input: Tuple) -> Tuple:
        module_name = self.name()

        if module_name == "GLMEmbedding":
            output = self.wrapper_module_.forward(input[0])
            if input[-1].gradient_checkpoint_ != "none":
                output = output.requires_grad_(True)
            return (output,) + input[1:]
        elif module_name == "GLMLayerNorm":
            output = self.wrapper_module_.forward(input[0])
            return (output,) + input[1:]
        elif module_name == "GLMBlock":
            outputs = CHECKPOINT_CLASSES[input[-1].gradient_checkpoint_](
                self.wrapper_module_.forward, *input
            )
            if len(outputs) > 1:
                self.router_probs_ = outputs[1:]
            return (outputs[0],) + input[1:]
        else:
            raise f"module invalid:{module_name}"


class GLMForCausalLM(LLMForCausalLM):
    def __init__(self, config: GLMConfig) -> None:
        self.config_ = config
        self.padding_idx_ = config.pad_token_id_
        self.vocab_size_ = config.vocab_size_

        # Embedding layer.
        self.embed_tokens_ = GLMEmbedding(config)
        # Rotary Position Embedding.
        self.rotary_emb_layer: torch.tensor = None
        # Encoder(Decoder) layers.
        self.layers_: List[GLMBlock] = []
        # Final layer norm.
        if self.config_.post_layer_norm:
            self.final_layernorm_ = GLMLayerNorm(config)
        # Output layer.
        self.lm_head_ = torch.nn.Linear(
            config.dim_,
            config.vocab_size_,
            bias=config.add_bias_linear,
            dtype=config.dtype_,
            device=config.device_,
        )

    def decoder_stack(self) -> List[LLMDecoder]:
        return self.layers_

    def sequential_module(self) -> OrderedDict:
        seq_module = OrderedDict()

        seq_module.update({"embedding": GLMSequentialWrapper(self.embed_tokens_)})
        seq_module.move_to_end("embedding")

        for index, layer in enumerate(self.decoder_stack()):
            layer_name = f"layer{index}"
            seq_module.update({layer_name: GLMSequentialWrapper(layer)})
            seq_module.move_to_end(layer_name)

        if self.config_.post_layer_norm:
            seq_module.update({"norm": GLMSequentialWrapper(self.final_layernorm_)})
            seq_module.move_to_end("norm")

        return seq_module

    def causal_mask(
        self,
        input_tokens: torch.Tensor,
        additional_mask: List[Masks] = None,
        diagonal: int = 1,
    ) -> torch.Tensor:

        return prepare_4d_causal_attention_mask(
            input_tokens=input_tokens,
            n_heads=1,
            additional_mask=additional_mask,
            diagonal=diagonal,
            dtype=self.config_.dtype_,
            device=self.config_.device_,
        )

    @staticmethod
    def from_pretrained(
        llm_model,
        attn_impl: str = "eager",
        use_sliding_window: bool = False,
        device: str = get_backend().default_device_name(),
    ):
        assert not use_sliding_window, "ChatGLM model does not support SWA."
        assert (
            attn_impl == "eager" or "flash_attn"
        ), "ChatGLM only supports eager or flash attention."

        # Get the config from LLM model and input args.
        llm_config = llm_model.config
        config = GLMConfig(
            # LLM model args.
            name_or_path_=llm_config._name_or_path,
            device_=device,
            dim_=llm_config.hidden_size,
            n_heads_=llm_config.num_attention_heads,
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
            attention_dropout=llm_config.attention_dropout,
            fp32_residual_connection=llm_config.fp32_residual_connection,
            apply_query_key_layer_scaling=llm_config.apply_query_key_layer_scaling,
            kv_channels=llm_config.kv_channels,
            multi_query_attention=llm_config.multi_query_attention,
            multi_query_group_num=llm_config.multi_query_group_num,
            attention_softmax_in_fp32=llm_config.attention_softmax_in_fp32,
            original_rope=llm_config.original_rope,
            add_bias_linear=llm_config.add_bias_linear,
            padded_vocab_size=llm_config.padded_vocab_size,
        )

        model = GLMForCausalLM(config=config)
        llm_model.requires_grad_(False)

        # =============================================
        #               Embedding Layer
        #   Load weights from the LLM model directly.
        # =============================================
        copy_parameters(
            llm_model.transformer.embedding.word_embeddings,
            model.embed_tokens_.embed_tokens,
        )

        # =============================================
        #       Rotary Position Embedding Layer
        # =============================================
        rotary_dim = (
            config.dim_ // config.n_heads_
            if config.kv_channels is None
            else config.kv_channels
        )
        model.rotary_emb_layer = RotaryEmbedding(
            dim=rotary_dim // 2,
            original_impl=config.original_rope,
            device=device,
            dtype=config.dtype_,
        )

        # =============================================
        #            Encoder(Decoder) Layers
        # =============================================
        for idx, layer in enumerate(llm_model.transformer.encoder.layers):
            # Get self-attention layer.
            self_attention = GLMSelfAttention(
                qkv_layer=layer.self_attention.query_key_value,
                dense_layer=layer.self_attention.dense,
                rotary_pos_emb=model.rotary_emb_layer(config.max_seq_len_),
                config=config,
                layer_idx=idx,
            )
            # Get MLP layer.
            mlp = FeedForward(
                GLMMLP(layer.mlp.dense_h_to_4h, layer.mlp.dense_4h_to_h, config=config)
            )
            # Create a transformer block.
            encoder = GLMBlock(self_attention, mlp, config)
            encoder.input_layernorm.load_weight(layer.input_layernorm.weight)
            encoder.post_layernorm.load_weight(layer.post_attention_layernorm.weight)
            model.layers_.append(encoder)

        # =============================================
        #               Final Layer Norm
        # =============================================
        if config.post_layer_norm:
            model.final_layernorm_.load_weight(
                llm_model.transformer.encoder.final_layernorm.weight
            )

        # =============================================
        #                   Output Layer
        #   Load weights from the LLM model directly.
        # =============================================
        copy_parameters(llm_model.transformer.output_layer, model.lm_head_)

        return model
