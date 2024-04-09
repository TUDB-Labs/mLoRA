from mlora.common import (
    _flash_attn_available,
    _xformers_available,
    prepare_4d_causal_attention_mask,
    scaled_dot_product_attention,
    xformers_attention,
    precompute_rope_angle,
    apply_rotary_emb,
    get_unpad_data,
    repeat_kv,
    Masks,
    Linear,
    FeedForward,
    MultiLoraBatchData,
    CheckpointRecomputeFunction as CheckpointFunction,
    LLMModelArgs,
    LLMAttention,
    LLMFeedForward,
    LLMDecoder,
    LLMForCausalLM,
)
from mlora.backends import _backend, get_backend
from mlora.utils import copy_parameters

from typing import Tuple, Dict, List, Optional
from transformers.activations import ACT2FN
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers.models.llama.modeling_llama as modeling_llama

if _flash_attn_available:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input


@dataclass
class LlamaConfig(LLMModelArgs):
    rms_norm_eps_: float = 1e-6


# Multi-headed attention from 'Attention Is All You Need' paper.
class LlamaAttention(LLMAttention):
    def __init__(self, wq: nn.Module, wk: nn.Module, wv: nn.Module, wo: nn.Module,
                 layer_idx: int, args: LlamaConfig):
        super().__init__()
        # attention
        self.wq_: Linear = Linear(wq, args.device_)  # dim * dim
        self.wk_: Linear = Linear(wk, args.device_)  # dim * dim
        self.wv_: Linear = Linear(wv, args.device_)  # dim * dim
        self.wo_: Linear = Linear(wo, args.device_)  # dim * dim
        # cos and sin
        self.cos_, self.sin_ = precompute_rope_angle(
            args.dim_ // args.n_heads_, args.max_seq_len_, args.rope_theta_, args.device_)
        # config
        self.layer_id_ = layer_idx
        self.dim_ = args.dim_
        self.n_heads_ = args.n_heads_
        self.n_kv_heads_ = args.n_kv_heads_
        self.n_rep_ = self.n_heads_ // self.n_kv_heads_
        self.head_dim_ = args.dim_ // args.n_heads_
        self.dtype_ = args.dtype_
        self.is_causal_ = True

    def state_dict(self) -> Dict[str, Linear]:
        return {
            "q_proj": self.wq_,
            "k_proj": self.wk_,
            "v_proj": self.wv_,
            "o_proj": self.wo_,
        }

    def forward(self,
                hidden_states: torch.Tensor,
                input_args: MultiLoraBatchData,
                attention_mask: Optional[torch.Tensor] = None):
        batch_size, max_seq_len, _ = hidden_states.shape

        xq = self.wq_.forward(hidden_states, input_args)
        xk = self.wk_.forward(hidden_states, input_args)
        xv = self.wv_.forward(hidden_states, input_args)

        # conver shape to multi head
        xq = xq.view(batch_size, max_seq_len, self.n_heads_,
                     self.head_dim_).transpose(1, 2)
        xk = xk.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)
        xv = xv.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)

        # apply rotary embedding
        assert xq.dtype == xk.dtype
        xq, xk = apply_rotary_emb(xq, xk, max_seq_len, self.cos_, self.sin_)

        # for llama2 need to repeat the heads
        # before dim: batch_size, n_kv_head, seq_len, head_dim
        # after dim: batch_size, n_head, seq_len, head_dim
        xk = repeat_kv(xk, self.n_rep_)
        xv = repeat_kv(xv, self.n_rep_)

        attention_score = scaled_dot_product_attention(
            xq, xk, xv, attention_mask)

        attention_score = attention_score.reshape(batch_size, max_seq_len, -1)

        # get output attention score
        return self.wo_.forward(attention_score, input_args)


class LlamaXformersAttention(LlamaAttention):
    def __init__(self, wq: nn.Module, wk: nn.Module, wv: nn.Module, wo: nn.Module,
                 layer_idx: int, args: LlamaConfig):
        assert _xformers_available, "xFormers Attention is not available"
        super().__init__(wq, wk, wv, wo, layer_idx, args)

    def forward(self,
                hidden_states: torch.Tensor,
                input_args: MultiLoraBatchData,
                attention_mask: Optional[torch.Tensor] = None):
        batch_size, max_seq_len, _ = hidden_states.shape

        xq = self.wq_.forward(hidden_states, input_args)
        xk = self.wk_.forward(hidden_states, input_args)
        xv = self.wv_.forward(hidden_states, input_args)

        # conver shape to multi head
        xq = xq.view(batch_size, max_seq_len, self.n_heads_,
                     self.head_dim_).transpose(1, 2)
        xk = xk.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)
        xv = xv.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)

        # apply rotary embedding
        assert xq.dtype == xk.dtype
        xq, xk = apply_rotary_emb(xq, xk, max_seq_len, self.cos_, self.sin_)

        # for llama2 need to repeat the heads
        # before dim: batch_size, n_kv_head, seq_len, head_dim
        # after dim: batch_size, n_head, seq_len, head_dim
        xk = repeat_kv(xk, self.n_rep_)
        xv = repeat_kv(xv, self.n_rep_)

        attention_score = xformers_attention(
            xq, xk, xv, attention_mask)

        attention_score = attention_score.reshape(batch_size, max_seq_len, -1)

        # get output attention score
        return self.wo_.forward(attention_score, input_args)


class LlamaFlashAttention(LlamaAttention):
    def __init__(self, wq: nn.Module, wk: nn.Module, wv: nn.Module, wo: nn.Module,
                 layer_idx: int, args: LlamaConfig):
        assert _flash_attn_available, "Flash Attention is not available"
        super().__init__(wq, wk, wv, wo, layer_idx, args)

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
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
                softmax_scale=softmax_scale,
                causal=self.is_causal_,
            )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=self.is_causal_
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = get_unpad_data(
            attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len,
                              num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len,
                                num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len,
                                    self.n_heads_, head_dim), indices_k
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
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def forward(self,
                hidden_states: torch.Tensor,
                input_args: MultiLoraBatchData,
                attention_mask: Optional[torch.Tensor] = None):
        batch_size, max_seq_len, _ = hidden_states.shape

        xq = self.wq_.forward(hidden_states, input_args)
        xk = self.wk_.forward(hidden_states, input_args)
        xv = self.wv_.forward(hidden_states, input_args)

        # conver shape to multi head
        xq = xq.view(batch_size, max_seq_len, self.n_heads_,
                     self.head_dim_).transpose(1, 2)
        xk = xk.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)
        xv = xv.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)

        # apply rotary embedding
        assert xq.dtype == xk.dtype
        xq, xk = apply_rotary_emb(xq, xk, max_seq_len, self.cos_, self.sin_)

        input_dtype = xq.dtype
        if input_dtype == torch.float32:
            if _backend.is_bf16_supported():
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float16
            xq = xq.to(target_dtype)
            xk = xk.to(target_dtype)
            xv = xv.to(target_dtype)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            xq,
            xk,
            xv,
            attention_mask,
            max_seq_len,
        ).to(input_dtype)

        attn_output = attn_output.reshape(
            batch_size, max_seq_len, self.dim_).contiguous()
        attn_output = self.wo_.forward(attn_output, input_args)

        return attn_output


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "xformers": LlamaXformersAttention,
    "flash_attn": LlamaFlashAttention,
}


class LlamaMLP(LLMFeedForward):
    def __init__(self, w1: nn.Module, w2: nn.Module, w3: nn.Module, args: LlamaConfig) -> None:
        super().__init__()
        # feed forward
        self.w1_: Linear = Linear(w1, args.device_)
        self.w2_: Linear = Linear(w2, args.device_)
        self.w3_: Linear = Linear(w3, args.device_)
        self.act_ = ACT2FN["silu"]

    def state_dict(self) -> Dict[str, nn.Module]:
        return {
            "w1_proj": self.w1_,
            "w2_proj": self.w2_,
            "w3_proj": self.w3_,
        }

    def _batch_forward(self, data: torch.Tensor, input_args: MultiLoraBatchData) -> torch.Tensor:
        w1 = self.w1_.forward(data, input_args)
        w3 = self.w3_.forward(data, input_args)
        return self.w2_.forward(self.act_(w1) * w3, input_args)

    def _lora_forward(
            self, lora_name: str, act_fn: nn.Module, data: torch.Tensor) -> torch.Tensor:
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


class LlamaRMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.norm_eps_ = eps
        self.weight_ = weight

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        input_dtype = data.dtype
        v = data.to(torch.float32).pow(2).mean(-1, keepdim=True)
        data = data * torch.rsqrt(v + self.norm_eps_)

        return (self.weight_ * data).to(input_dtype)


class LlamaDecoderLayer(LLMDecoder):
    def __init__(self) -> None:
        super().__init__()
        self.layer_id_: int = None
        self.self_attn_: LlamaAttention = None
        self.mlp_: FeedForward = None
        self.input_layernorm_: LlamaRMSNorm = None
        self.post_attention_layernorm_: LlamaRMSNorm = None

    def state_dict(self) -> Dict[str, nn.Module]:
        linear_layers = self.self_attn_.state_dict()
        linear_layers.update(self.mlp_.state_dict())
        return linear_layers

    def forward(self,
                hidden_states: torch.Tensor,
                input_args: MultiLoraBatchData,
                attention_mask: Optional[torch.Tensor] = None,
                router_logits: Optional[List[List]] = None):
        residual = hidden_states
        hidden_states = self.input_layernorm_(hidden_states)
        # Self Attention
        hidden_states = self.self_attn_.forward(
            hidden_states, input_args, attention_mask)
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm_(hidden_states)
        hidden_states = self.mlp_.forward(
            hidden_states, input_args, router_logits)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaEmbedding(nn.Module):
    def __init__(self, embedding: torch.Tensor, pad_token: int):
        super().__init__()
        self.token_embedding_: torch.Tensor = embedding
        self.padding_idx_: int = pad_token

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        data = F.embedding(tokens, self.token_embedding_,
                           padding_idx=self.padding_idx_)
        return data


class LlamaSequentialWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.wrapper_module_ = module

    def name(self) -> str:
        return type(self.wrapper_module_).__name__

    def forward(self, input: Tuple) -> Tuple:
        module_name = self.name()

        if module_name == "LlamaEmbedding":
            output = self.wrapper_module_.forward(input[0])
            if input[-1]:
                output = output.requires_grad_(True)
            return (output, ) + input[1:]
        elif module_name == "LlamaRMSNorm":
            output = self.wrapper_module_.forward(input[0])
            return (output, ) + input[1:]
        elif module_name == "LlamaDecoderLayer":
            if input[-1]:
                output = CheckpointFunction.apply(
                    self.wrapper_module_.forward, *input[:-1])
            else:
                output = self.wrapper_module_.forward(*input[:-1])
            return (output, ) + input[1:]
        else:
            raise f"module invalid: {module_name}"


class LlamaForCausalLM(LLMForCausalLM):
    def __init__(self, config: LlamaConfig) -> None:
        self.config_ = config
        self.padding_idx_ = config.pad_token_id_
        self.vocab_size_ = config.vocab_size_
        self.embed_tokens_: LlamaEmbedding = None
        self.norm_: LlamaEmbedding = None
        self.lm_head_ = nn.Linear(config.dim_, config.vocab_size_, bias=False,
                                  dtype=config.dtype_, device=config.device_)
        self.layers_: List[LlamaDecoderLayer] = []

    def decoder_stack(self) -> List[LLMDecoder]:
        return self.layers_

    def sequential_module(self) -> OrderedDict:
        seq_module = OrderedDict()

        seq_module.update(
            {"embedding": LlamaSequentialWrapper(self.embed_tokens_)})
        seq_module.move_to_end("embedding")

        for index, layer in enumerate(self.layers_):
            layer_name = f"layer{index}"
            seq_module.update({layer_name: LlamaSequentialWrapper(layer)})
            seq_module.move_to_end(layer_name)

        seq_module.update(
            {"norm": LlamaSequentialWrapper(self.norm_)})
        seq_module.move_to_end("norm")

        return seq_module

    def causal_mask(self,
                    input_tokens: torch.Tensor,
                    additional_mask: List[Masks] = None,
                    multi_head: bool = False,
                    diagonal: int = 1) -> torch.Tensor:
        if multi_head:
            assert self.config_.attn_implementation_ == "xformers"
        else:
            assert self.config_.attn_implementation_ != "xformers"

        return prepare_4d_causal_attention_mask(input_tokens=input_tokens,
                                                n_heads=self.config_.n_heads_ if multi_head else 1,
                                                additional_mask=additional_mask, diagonal=diagonal,
                                                dtype=self.config_.dtype_, device=self.config_.device_)

    @staticmethod
    def from_pretrained(llm_model: modeling_llama.LlamaForCausalLM,
                        attn_impl: str = "eager",
                        use_sliding_window: bool = False,
                        device: str = get_backend().device_name() + ":0"):
        assert not use_sliding_window, "Llama model does not support SWA."
        llm_config: modeling_llama.LlamaConfig = llm_model.config
        llm_args = LlamaConfig(
            name_or_path_=llm_config.name_or_path,
            vocab_size_=llm_config.vocab_size,
            dim_=llm_config.hidden_size,
            n_layers_=llm_config.num_hidden_layers,
            n_heads_=llm_config.num_attention_heads,
            n_kv_heads_=llm_config.num_key_value_heads,
            rms_norm_eps_=llm_config.rms_norm_eps,
            max_seq_len_=llm_config.max_position_embeddings,
            rope_theta_=llm_config.rope_theta,
            pad_token_id_=llm_config.pad_token_id,
            attn_implementation_=attn_impl,
            device_=torch.device(device),
            dtype_=llm_model.dtype,
        )

        if llm_args.pad_token_id_ is None:
            llm_args.pad_token_id_ = -1

        model = LlamaForCausalLM(llm_args)
        llm_model.requires_grad_(False)
        model.embed_tokens_ = LlamaEmbedding(
            llm_model.model.embed_tokens.weight, llm_args.pad_token_id_)
        model.norm_ = LlamaRMSNorm(
            llm_model.model.norm.weight, llm_args.rms_norm_eps_)
        copy_parameters(llm_model.lm_head, model.lm_head_)

        for idx, layer in enumerate(llm_model.model.layers):
            decoder = LlamaDecoderLayer()
            decoder.layer_id_ = idx
            decoder.self_attn_ = LLAMA_ATTENTION_CLASSES[llm_args.attn_implementation_](
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
                idx,
                llm_args,
            )
            decoder.mlp_ = FeedForward(LlamaMLP(
                layer.mlp.gate_proj,
                layer.mlp.down_proj,
                layer.mlp.up_proj,
                llm_args,
            ))
            decoder.input_layernorm_ = LlamaRMSNorm(
                layer.input_layernorm.weight, llm_args.rms_norm_eps_)
            decoder.post_attention_layernorm_ = LlamaRMSNorm(
                layer.post_attention_layernorm.weight, llm_args.rms_norm_eps_)
            model.layers_.append(decoder)

        return model
