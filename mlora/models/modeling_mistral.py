from mlora.models.modeling_llama import (
    LlamaConfig,
    LlamaAttention,
    LlamaXformersAttention,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaEmbedding,
    LlamaForCausalLM,
)
from mlora.common import (
    _flash_attn_available,
    apply_rotary_emb,
    get_unpad_data,
    repeat_kv,
    FeedForward,
    MultiLoraBatchData,
)
from mlora.backends import _backend, get_backend
from mlora.utils import copy_parameters

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import transformers.models.qwen2.modeling_qwen2 as modeling_qwen2
import transformers.models.mistral.modeling_mistral as modeling_mistral

if _flash_attn_available:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input


@dataclass
class MistralConfig(LlamaConfig):
    use_sliding_window_: bool = False
    max_window_layers_: int = None
    sliding_window_: int = None


class MistralFlashAttention(LlamaAttention):
    def __init__(self, wq: nn.Module, wk: nn.Module, wv: nn.Module, wo: nn.Module, args: MistralConfig):
        assert _flash_attn_available, "Flash Attention is not available"
        super().__init__(wq, wk, wv, wo, args)
        # Qwen2
        self.use_sliding_window_ = args.use_sliding_window_
        self.max_window_layers_ = args.max_window_layers_
        # Mistral and Qwen2
        self.sliding_window_ = args.sliding_window_

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
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
            else:
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
                    window_size=(self.sliding_window_,
                                 self.sliding_window_),
                )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal_,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=self.is_causal_,
                    window_size=(self.sliding_window_,
                                 self.sliding_window_),
                )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:,
                                            attention_mask_num_tokens - kv_seq_len:]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = get_unpad_data(
            attention_mask)

        key_layer = index_first_axis(key_layer.reshape(
            batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(
            batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len,
                                    num_heads, head_dim), indices_k
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

        kv_seq_len = xk.shape[-2]

        # apply rotary embedding
        assert xq.dtype == xk.dtype
        xq, xk = apply_rotary_emb(xq, xk, max_seq_len, self.cos_, self.sin_)

        use_sliding_windows = (
            (self.use_sliding_window_ is None or self.use_sliding_window_)
            and (self.sliding_window_ is not None and kv_seq_len > self.sliding_window_)
            and (self.max_window_layers_ is None or self.layer_id_ < self.max_window_layers_)
        )

        # for llama2 need to repeat the heads
        # before dim: batch_size, n_kv_head, seq_len, head_dim
        # after dim: batch_size, n_head, seq_len, head_dim
        xk = repeat_kv(xk, self.n_rep_)
        xv = repeat_kv(xv, self.n_rep_)

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
            use_sliding_windows=use_sliding_windows,
        ).to(input_dtype)

        attn_output = attn_output.reshape(
            batch_size, max_seq_len, self.dim_).contiguous()
        attn_output = self.wo_.forward(attn_output, input_args)

        return attn_output


MISTRAL_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "xformers": LlamaXformersAttention,
    "flash_attn": MistralFlashAttention,
}


class MistralForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)

    @staticmethod
    def from_pretrained(llm_model: modeling_mistral.MistralForCausalLM,
                        attn_impl: str = "eager",
                        use_sliding_window: bool = False,
                        device: str = get_backend().device_name() + ":0"):
        llm_config: modeling_mistral.MistralConfig = llm_model.config
        llm_args = MistralConfig(
            name_or_path_=llm_config.name_or_path,
            vocab_size_=llm_config.vocab_size,
            dim_=llm_config.hidden_size,
            intermediate_=llm_config.intermediate_size,
            n_layers_=llm_config.num_hidden_layers,
            n_heads_=llm_config.num_attention_heads,
            n_kv_heads_=llm_config.num_key_value_heads,
            hidden_act_=llm_config.hidden_act,
            rms_norm_eps_=llm_config.rms_norm_eps,
            max_seq_len_=llm_config.max_position_embeddings,
            rope_theta_=llm_config.rope_theta,
            pad_token_id_=llm_config.pad_token_id,
            attn_implementation_=attn_impl,
            use_sliding_window_=use_sliding_window,
            sliding_window_=llm_config.sliding_window,
            device_=torch.device(device),
            dtype_=llm_model.dtype,
        )

        if use_sliding_window and attn_impl != "flash_attn":
            raise ValueError(
                f"Can not use sliding window attention with {attn_impl} attention.")

        # compatible with qwen2
        if isinstance(llm_config, modeling_qwen2.Qwen2Config):
            llm_args.max_window_layers_ = llm_config.max_window_layers

        if llm_args.pad_token_id_ is None:
            llm_args.pad_token_id_ = -1

        model = MistralForCausalLM(llm_args)
        llm_model.requires_grad_(False)
        model.embed_tokens_ = LlamaEmbedding(
            llm_model.model.embed_tokens.weight, llm_args.pad_token_id_)
        model.norm_ = LlamaRMSNorm(
            llm_model.model.norm.weight, llm_args.rms_norm_eps_)
        copy_parameters(llm_model.lm_head, model.lm_head_)

        for idx, layer in enumerate(llm_model.model.layers):
            decoder = LlamaDecoderLayer(idx)
            decoder.self_attn_ = MISTRAL_ATTENTION_CLASSES[llm_args.attn_implementation_](
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
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
