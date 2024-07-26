from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.gemma2 import modeling_gemma2
from transformers.models.gemma2.modeling_gemma2 import apply_rotary_pos_emb, repeat_kv
from transformers.utils import is_flash_attn_2_available

from mlora.backends import backend
from mlora.common import (
    Cache,
    FeedForward,
    Linear,
    LLMAttention,
    LLMDecoder,
    LLMForCausalLM,
    LLMModelConfig,
    LLMModelInput,
    flash_attention_forward,
    prepare_4d_causal_attention_mask,
)
from mlora.models.modeling_gemma import GemmaEmbedding, GemmaRMSNorm
from mlora.models.modeling_llama import LlamaMLP
from mlora.utils import copy_parameters, is_package_available


@dataclass
class Gemma2Config(LLMModelConfig):
    rms_norm_eps_: float = 1e-6
    attn_logit_softcapping_: float = 50.0
    final_logit_softcapping_: float = 30.0
    query_pre_attn_scalar_: int = 224
    use_sliding_window_: bool = False
    sliding_window_: int = 4096


class Gemma2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        self.inv_freq.to(x.device)
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Multi-headed attention from 'Attention Is All You Need' paper.
class Gemma2Attention(LLMAttention):
    def __init__(
        self,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        o_proj: nn.Module,
        layer_idx: int,
        config: Gemma2Config,
    ):
        super().__init__()
        # attention
        self.q_proj_: Linear = Linear(q_proj, config.device_)
        self.k_proj_: Linear = Linear(k_proj, config.device_)
        self.v_proj_: Linear = Linear(v_proj, config.device_)
        self.o_proj_: Linear = Linear(o_proj, config.device_)
        # config
        self.layer_idx_ = layer_idx
        self.config_ = config
        self.dim_ = config.dim_
        self.n_heads_ = config.n_heads_
        self.n_kv_heads_ = config.n_kv_heads_
        self.n_rep_ = self.n_heads_ // self.n_kv_heads_
        self.head_dim_ = config.head_dim_
        self.dtype_ = config.dtype_
        self.is_causal_ = True

        self.scaling_ = config.query_pre_attn_scalar_**-0.5
        self.sliding_window_ = (
            config.sliding_window_
            if config.use_sliding_window_ and not bool(layer_idx % 2)
            else None
        )

    def state_dict(self) -> Dict[str, Linear]:
        return {
            "q_proj": self.q_proj_,
            "k_proj": self.k_proj_,
            "v_proj": self.v_proj_,
            "o_proj": self.o_proj_,
        }

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj_(hidden_states, input_args)
        key_states = self.k_proj_(hidden_states, input_args)
        value_states = self.v_proj_(hidden_states, input_args)

        query_states = query_states.view(
            bsz, q_len, self.n_heads_, self.head_dim_
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)

        cos, sin = rotary_emb
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "sliding_window": self.sliding_window_,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx_, cache_kwargs
            )

        key_states = repeat_kv(key_states, self.n_rep_)
        value_states = repeat_kv(value_states, self.n_rep_)

        attn_weights = (
            torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling_
        )

        if self.config_.attn_logit_softcapping_ is not None:
            attn_weights = attn_weights / self.config_.attn_logit_softcapping_
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.config_.attn_logit_softcapping_

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(bsz, q_len, -1)
        return self.o_proj_(attn_output, input_args)


class Gemma2FlashAttention2(Gemma2Attention):
    def __init__(
        self,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        o_proj: nn.Module,
        layer_idx: int,
        config: Gemma2Config,
    ):
        assert is_flash_attn_2_available(), "Flash Attention is not available"
        super().__init__(q_proj, k_proj, v_proj, o_proj, layer_idx, config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj_(hidden_states, input_args)
        key_states = self.k_proj_(hidden_states, input_args)
        value_states = self.v_proj_(hidden_states, input_args)

        query_states = query_states.view(
            bsz, q_len, self.n_heads_, self.head_dim_
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.n_kv_heads_, self.head_dim_
        ).transpose(1, 2)

        cos, sin = rotary_emb
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "sliding_window": self.sliding_window_,
                "cache_position": cache_position,
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx_, cache_kwargs
            )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if backend.is_bf16_supported():
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float16
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            is_causal=self.is_causal_,
            softmax_scale=self.scaling_,
            sliding_window=(
                self.sliding_window_ if self.config_.use_sliding_window_ else None
            ),
            softcap=(
                self.config_.attn_logit_softcapping_
                if is_package_available("flash_attn", "2.6.0")
                else None
            ),
        ).to(input_dtype)

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj_(attn_output, input_args)

        return attn_output


GEMMA2_ATTENTION_CLASSES = {
    "eager": Gemma2Attention,
    "flash_attn": Gemma2FlashAttention2,
}


class Gemma2DecoderLayer(LLMDecoder):
    def __init__(self, layer_idx: int, config: Gemma2Config) -> None:
        super().__init__()
        self.layer_id_: int = layer_idx
        self.self_attn_: Gemma2Attention = None
        self.mlp_: FeedForward = None
        self.input_layernorm_: GemmaRMSNorm = None
        self.post_attention_layernorm_: GemmaRMSNorm = None

        self.config_ = config
        self.is_sliding_ = not bool(layer_idx % 2)
        self.pre_feedforward_layernorm_: GemmaRMSNorm = None
        self.post_feedforward_layernorm_: GemmaRMSNorm = None
        self.sliding_window_ = config.sliding_window_

    def state_dict(self) -> Dict[str, nn.Module]:
        linear_layers = self.self_attn_.state_dict()
        linear_layers.update(self.mlp_.state_dict())
        return linear_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
    ):
        if (
            self.config_.attn_implementation_ != "flash_attn"
            and self.config_.use_sliding_window_
            and self.is_sliding_
            and attention_mask is not None
        ):
            min_dtype = torch.finfo(hidden_states.dtype).min
            sliding_window_mask = torch.tril(
                torch.ones_like(attention_mask, dtype=torch.bool),
                diagonal=-self.sliding_window_,
            )
            attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask)
            if attention_mask.shape[-1] <= 1:  # when decoding
                attention_mask = attention_mask[:, :, :, -self.sliding_window_ :]

        residual = hidden_states

        hidden_states = self.input_layernorm_(hidden_states)

        hidden_states = self.self_attn_.forward(
            hidden_states,
            input_args,
            rotary_emb,
            attention_mask,
            cache_position,
            past_key_value,
        )
        hidden_states = self.post_attention_layernorm_(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm_(hidden_states)
        hidden_states, router_logits = self.mlp_.forward(hidden_states, input_args)
        hidden_states = self.post_feedforward_layernorm_(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, *router_logits


class Gemma2OutputLayer(nn.Module):
    def __init__(self, config: Gemma2Config):
        super().__init__()
        self.lm_head_ = nn.Linear(
            config.dim_,
            config.vocab_size_,
            bias=False,
            dtype=config.dtype_,
            device=config.device_,
        )
        self.final_logit_softcapping_ = config.final_logit_softcapping_

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head_(hidden_states)
        if self.final_logit_softcapping_ is not None:
            logits = logits / self.final_logit_softcapping_
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping_
        return logits


class Gemma2ForCausalLM(LLMForCausalLM):
    def __init__(self, config: Gemma2Config) -> None:
        super().__init__()
        self.config_ = config
        self.padding_idx_ = config.pad_token_id_
        self.vocab_size_ = config.vocab_size_
        self.embed_tokens_: GemmaEmbedding = None
        self.norm_: GemmaRMSNorm = None
        self.rotary_emb_ = Gemma2RotaryEmbedding(
            config.head_dim_,
            max_position_embeddings=config.max_seq_len_,
            base=config.rope_theta_,
            device=config.device_,
        )
        self.lm_head_ = Gemma2OutputLayer(config)
        self.layers_: List[Gemma2DecoderLayer] = []

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens_(input_ids)

    def rotary_embed(
        self, input_tensor: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rotary_emb_(input_tensor, position_ids)

    def decoder_stack(self) -> List[LLMDecoder]:
        return self.layers_

    def norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm_(hidden_states)

    def causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[Cache],
    ) -> torch.Tensor:

        return prepare_4d_causal_attention_mask(
            attention_mask,
            input_tensor,
            cache_position,
            past_key_values,
        )

    def cache_implementation(self) -> str:
        if self.config_.use_sliding_window_ and self.config_.sliding_window_:
            return "hybrid"
        else:
            return "dynamic"

    def model_config(self) -> Gemma2Config:
        return self.config_

    @staticmethod
    def from_pretrained(
        llm_model: modeling_gemma2.Gemma2PreTrainedModel,
        attn_impl: str = "eager",
        use_sliding_window: bool = False,
        device: str = backend.default_device_name(),
    ):
        llm_config: modeling_gemma2.Gemma2Config = llm_model.config
        model_config = Gemma2Config(
            name_or_path_=llm_config.name_or_path,
            vocab_size_=llm_config.vocab_size,
            dim_=llm_config.hidden_size,
            head_dim_=llm_config.head_dim,
            intermediate_=llm_config.intermediate_size,
            n_layers_=llm_config.num_hidden_layers,
            n_heads_=llm_config.num_attention_heads,
            n_kv_heads_=llm_config.num_key_value_heads,
            hidden_act_=llm_config.hidden_activation,
            rms_norm_eps_=llm_config.rms_norm_eps,
            max_seq_len_=llm_config.max_position_embeddings,
            rope_theta_=llm_config.rope_theta,
            attn_logit_softcapping_=llm_config.attn_logit_softcapping,
            final_logit_softcapping_=llm_config.final_logit_softcapping,
            query_pre_attn_scalar_=llm_config.query_pre_attn_scalar,
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

        if model_config.pad_token_id_ is None:
            model_config.pad_token_id_ = -1

        model = Gemma2ForCausalLM(model_config)
        llm_model.requires_grad_(False)
        model.embed_tokens_ = GemmaEmbedding(
            llm_model.model.embed_tokens.weight,
            model_config.pad_token_id_,
            model_config.dim_**0.5,
        )
        model.norm_ = GemmaRMSNorm(
            llm_model.model.norm.weight, model_config.rms_norm_eps_
        )
        copy_parameters(llm_model.lm_head, model.lm_head_.lm_head_)

        for layer_idx, layer in enumerate(llm_model.model.layers):
            decoder = Gemma2DecoderLayer(layer_idx, model_config)
            decoder.self_attn_ = GEMMA2_ATTENTION_CLASSES[
                model_config.attn_implementation_
            ](
                layer.self_attn.q_proj,
                layer.self_attn.k_proj,
                layer.self_attn.v_proj,
                layer.self_attn.o_proj,
                layer_idx,
                model_config,
            )
            decoder.mlp_ = FeedForward(
                LlamaMLP(
                    layer.mlp.gate_proj,
                    layer.mlp.down_proj,
                    layer.mlp.up_proj,
                    model_config,
                )
            )
            decoder.input_layernorm_ = GemmaRMSNorm(
                layer.input_layernorm.weight, model_config.rms_norm_eps_
            )
            decoder.post_attention_layernorm_ = GemmaRMSNorm(
                layer.post_attention_layernorm.weight, model_config.rms_norm_eps_
            )
            decoder.pre_feedforward_layernorm_ = GemmaRMSNorm(
                layer.pre_feedforward_layernorm.weight, model_config.rms_norm_eps_
            )
            decoder.post_feedforward_layernorm_ = GemmaRMSNorm(
                layer.post_feedforward_layernorm.weight, model_config.rms_norm_eps_
            )
            model.layers_.append(decoder)

        return model
