import inspect
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers.models.mistral import modeling_mistral
from transformers.models.qwen2 import modeling_qwen2
from transformers.utils import is_flash_attn_2_available

from mlora.backends import _backend, get_backend
from mlora.common import Cache, FeedForward, LLMModelInput
from mlora.models.modeling_llama import (
    LlamaAttention,
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaEmbedding,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaRMSNorm,
    _get_unpad_data,
    apply_rotary_pos_emb,
    repeat_kv,
)
from mlora.utils import copy_parameters

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

    _flash_supports_window_size = "window_size" in list(
        inspect.signature(flash_attn_func).parameters
    )


@dataclass
class MistralConfig(LlamaConfig):
    use_sliding_window_: bool = False
    max_window_layers_: int = None
    sliding_window_: int = None


class MistralFlashAttention(LlamaAttention):
    def __init__(
        self,
        wq: nn.Module,
        wk: nn.Module,
        wv: nn.Module,
        wo: nn.Module,
        idx: int,
        args: MistralConfig,
    ):
        assert is_flash_attn_2_available(), "Flash Attention is not available"
        super().__init__(wq, wk, wv, wo, idx, args)
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
        causal = self.is_causal_

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (
                query_states,
                key_states,
                value_states,
                indices_q,
                cu_seq_lens,
                max_seq_lens,
            ) = self._upad_input(
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
                    causal=causal,
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
                    causal=causal,
                    window_size=(
                        self.sliding_window_,
                        self.sliding_window_,
                    ),
                )

            attn_output = pad_input(
                attn_output_unpad, indices_q, batch_size, query_length
            )
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(
                        self.sliding_window_,
                        self.sliding_window_,
                    ),
                )

        return attn_output

    def _upad_input(
        self, query_layer, key_layer, value_layer, attention_mask, query_length
    ):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
        )

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim),
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
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
    ):
        batch_size, max_seq_len, _ = hidden_states.shape

        xq = self.wq_.forward(hidden_states, input_args)
        xk = self.wk_.forward(hidden_states, input_args)
        xv = self.wv_.forward(hidden_states, input_args)

        # conver shape to multi head
        xq = xq.view(batch_size, max_seq_len, self.n_heads_, self.head_dim_).transpose(
            1, 2
        )
        xk = xk.view(
            batch_size, max_seq_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)
        xv = xv.view(
            batch_size, max_seq_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)

        kv_seq_len = xk.shape[-2]
        if past_key_value is not None:
            kv_seq_len += cache_position[0]

        # apply rotary embedding
        cos, sin = self.rotary_emb_(xv, cache_position.unsqueeze(0))
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        use_sliding_windows = (
            _flash_supports_window_size
            and (self.use_sliding_window_ is None or self.use_sliding_window_)
            and (self.sliding_window_ is not None and kv_seq_len > self.sliding_window_)
            and (
                self.max_window_layers_ is None
                or self.layer_id_ < self.max_window_layers_
            )
        )

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx_) > 0
            if (
                self.sliding_window_ is not None
                and kv_seq_len > self.sliding_window_
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.sliding_window_

                past_key = past_key_value[self.layer_idx_][0]
                past_value = past_key_value[self.layer_idx_][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.sliding_window_ - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.sliding_window - 1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat(
                        [attention_mask, torch.ones_like(attention_mask[:, -1:])],
                        dim=-1,
                    )

            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            xk, xv = past_key_value.update(xk, xv, self.layer_idx_, cache_kwargs)

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
            batch_size, max_seq_len, self.dim_
        ).contiguous()
        attn_output = self.wo_.forward(attn_output, input_args)

        return attn_output


MISTRAL_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attn": MistralFlashAttention,
}


class MistralForCausalLM(LlamaForCausalLM):
    def __init__(self, config: MistralConfig) -> None:
        super().__init__(config)

    @staticmethod
    def from_pretrained(
        llm_model: modeling_mistral.MistralForCausalLM,
        attn_impl: str = "eager",
        use_sliding_window: bool = False,
        device: str = get_backend().default_device_name(),
    ):
        llm_config: modeling_mistral.MistralConfig = llm_model.config
        llm_args = MistralConfig(
            name_or_path_=llm_config.name_or_path,
            vocab_size_=llm_config.vocab_size,
            dim_=llm_config.hidden_size,
            head_dim_=llm_config.hidden_size // llm_config.num_attention_heads,
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
                f"Can not use sliding window attention with {attn_impl} attention."
            )

        # compatible with qwen2
        if isinstance(llm_config, modeling_qwen2.Qwen2Config):
            llm_args.max_window_layers_ = llm_config.max_window_layers

        if llm_args.pad_token_id_ is None:
            llm_args.pad_token_id_ = -1

        model = MistralForCausalLM(llm_args)
        llm_model.requires_grad_(False)
        model.embed_tokens_ = LlamaEmbedding(
            llm_model.model.embed_tokens.weight, llm_args.pad_token_id_
        )
        model.norm_ = LlamaRMSNorm(llm_model.model.norm.weight, llm_args.rms_norm_eps_)
        copy_parameters(llm_model.lm_head, model.lm_head_)

        for idx, layer in enumerate(llm_model.model.layers):
            decoder = LlamaDecoderLayer(idx)
            decoder.self_attn_ = MISTRAL_ATTENTION_CLASSES[
                llm_args.attn_implementation_
            ](
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
                idx,
                llm_args,
            )
            decoder.mlp_ = FeedForward(
                LlamaMLP(
                    layer.mlp.gate_proj,
                    layer.mlp.down_proj,
                    layer.mlp.up_proj,
                    llm_args,
                )
            )
            decoder.input_layernorm_ = LlamaRMSNorm(
                layer.input_layernorm.weight, llm_args.rms_norm_eps_
            )
            decoder.post_attention_layernorm_ = LlamaRMSNorm(
                layer.post_attention_layernorm.weight, llm_args.rms_norm_eps_
            )
            model.layers_.append(decoder)

        return model
