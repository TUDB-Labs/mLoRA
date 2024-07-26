from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.utils import is_flash_attn_2_available

from mlora.backends import backend
from mlora.common import (
    ROPE_INIT_FUNCTIONS,
    Cache,
    FeedForward,
    Linear,
    LLMAttention,
    LLMDecoder,
    LLMFeedForward,
    LLMForCausalLM,
    LLMModelConfig,
    LLMModelInput,
    eager_attention_forward,
    flash_attention_forward,
    prepare_4d_causal_attention_mask,
)
from mlora.common.mix_lora import _mixtral_slice_tensor
from mlora.utils import copy_parameters


@dataclass
class LlamaConfig(LLMModelConfig):
    rms_norm_eps_: float = 1e-6
    rope_scaling_: Optional[Dict[str, Any]] = None


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        config: Optional[LlamaConfig],
        scaling_factor=1.0,
        rope_type="default",
    ):
        super().__init__()
        self.rope_kwargs = {
            "rope_type": rope_type,
            "factor": scaling_factor,
            "dim": config.head_dim_,
            "base": config.rope_theta_,
            "max_position_embeddings": config.max_seq_len_,
        }
        if config is None:
            self.rope_type = rope_type
            self.max_seq_len_cached = config.max_seq_len_
            self.original_max_seq_len = config.max_seq_len_
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling_ is not None:
                self.rope_type = config.rope_scaling_.get(
                    "rope_type", config.rope_scaling_.get("type")
                )
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_seq_len_
            self.original_max_seq_len = config.max_seq_len_

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, config.device_, **self.rope_kwargs
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer(
                "inv_freq", inv_freq, persistent=False
            )  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if (
            seq_len < self.original_max_seq_len
            and self.max_seq_len_cached > self.original_max_seq_len
        ):  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
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

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Multi-headed attention from 'Attention Is All You Need' paper.
class LlamaAttention(LLMAttention):
    def __init__(
        self,
        wq: nn.Module,
        wk: nn.Module,
        wv: nn.Module,
        wo: nn.Module,
        idx: int,
        args: LlamaConfig,
    ):
        super().__init__()
        # attention
        self.wq_: Linear = Linear(wq, args.device_)  # dim * dim
        self.wk_: Linear = Linear(wk, args.device_)  # dim * dim
        self.wv_: Linear = Linear(wv, args.device_)  # dim * dim
        self.wo_: Linear = Linear(wo, args.device_)  # dim * dim
        # config
        self.layer_idx_ = idx
        self.dim_ = args.dim_
        self.n_heads_ = args.n_heads_
        self.n_kv_heads_ = args.n_kv_heads_
        self.n_rep_ = self.n_heads_ // self.n_kv_heads_
        self.head_dim_ = args.head_dim_
        self.dtype_ = args.dtype_
        self.is_causal_ = True

    def state_dict(self) -> Dict[str, Linear]:
        return {
            "q_proj": self.wq_,
            "k_proj": self.wk_,
            "v_proj": self.wv_,
            "o_proj": self.wo_,
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

        # apply rotary embedding
        cos, sin = rotary_emb
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            xk, xv = past_key_value.update(xk, xv, self.layer_idx_, cache_kwargs)

        # for llama2 need to repeat the heads
        # before dim: batch_size, n_kv_head, seq_len, head_dim
        # after dim: batch_size, n_head, seq_len, head_dim
        xk = repeat_kv(xk, self.n_rep_)
        xv = repeat_kv(xv, self.n_rep_)

        attention_score = eager_attention_forward(xq, xk, xv, attention_mask)
        attention_score = attention_score.reshape(batch_size, max_seq_len, -1)

        # get output attention score
        return self.wo_.forward(attention_score, input_args)


class LlamaFlashAttention(LlamaAttention):
    def __init__(
        self,
        wq: nn.Module,
        wk: nn.Module,
        wv: nn.Module,
        wo: nn.Module,
        idx: int,
        args: LlamaConfig,
    ):
        assert is_flash_attn_2_available(), "Flash Attention is not available"
        super().__init__(wq, wk, wv, wo, idx, args)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
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

        # apply rotary embedding
        cos, sin = rotary_emb
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }
            xk, xv = past_key_value.update(xk, xv, self.layer_idx_, cache_kwargs)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        input_dtype = xq.dtype
        if input_dtype == torch.float32:
            if backend.is_bf16_supported():
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float16
            xq = xq.to(target_dtype)
            xk = xk.to(target_dtype)
            xv = xv.to(target_dtype)

        attn_output = flash_attention_forward(
            xq,
            xk,
            xv,
            attention_mask,
            max_seq_len,
            is_causal=self.is_causal_,
        ).to(input_dtype)

        attn_output = attn_output.reshape(batch_size, max_seq_len, -1).contiguous()
        attn_output = self.wo_.forward(attn_output, input_args)

        return attn_output


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attn": LlamaFlashAttention,
}


class LlamaMLP(LLMFeedForward):
    def __init__(
        self, w1: nn.Module, w2: nn.Module, w3: nn.Module, args: LlamaConfig
    ) -> None:
        super().__init__()
        # feed forward
        self.w1_: Linear = Linear(w1, args.device_)
        self.w2_: Linear = Linear(w2, args.device_)
        self.w3_: Linear = Linear(w3, args.device_)
        self.act_ = ACT2FN[args.hidden_act_]

    def state_dict(self) -> Dict[str, nn.Module]:
        return {
            "gate_proj": self.w1_,
            "down_proj": self.w2_,
            "up_proj": self.w3_,
        }

    def _batch_forward(
        self, data: torch.Tensor, input_args: LLMModelInput
    ) -> torch.Tensor:
        w1 = self.w1_.forward(data, input_args)
        w3 = self.w3_.forward(data, input_args)
        return self.w2_.forward(self.act_(w1) * w3, input_args)

    def _lora_forward(
        self, lora_name: str, act_fn: nn.Module, data: torch.Tensor
    ) -> torch.Tensor:
        # Applying LoRA weights to FFN weights
        if lora_name in self.w1_.loras_:
            w1 = self.w1_.loras_[lora_name].forward(
                self.w1_.base_layer_.forward(data), data
            )
        else:
            w1 = self.w1_.base_layer_.forward(data)

        if lora_name in self.w3_.loras_:
            w3 = self.w3_.loras_[lora_name].forward(
                self.w3_.base_layer_.forward(data), data
            )
        else:
            w3 = self.w3_.base_layer_.forward(data)

        act_result = act_fn(w1) * w3
        if lora_name in self.w2_.loras_:
            return self.w2_.loras_[lora_name].forward(
                self.w2_.base_layer_.forward(act_result), act_result
            )
        else:
            return self.w2_.base_layer_.forward(act_result)

    def _mixlora_forward(
        self, moe_name, act_fn, expert_mask, hidden_states, input_dtype
    ):
        common_w1 = self.w1_.base_layer_.forward(hidden_states.to(input_dtype)).to(
            hidden_states.dtype
        )
        common_w3 = self.w3_.base_layer_.forward(hidden_states.to(input_dtype)).to(
            hidden_states.dtype
        )
        final_expert_states = []
        for expert_idx in range(expert_mask.shape[0]):
            _, top_x = torch.where(expert_mask[expert_idx])

            lora_name = f"moe.{moe_name}.experts.{expert_idx}"
            if lora_name in self.w1_.loras_:
                lora_data = _mixtral_slice_tensor(hidden_states, top_x, input_dtype)
                w1 = self.w1_.loras_[lora_name].forward(
                    _mixtral_slice_tensor(common_w1, top_x, input_dtype), lora_data
                )
            else:
                lora_data = None
                w1 = _mixtral_slice_tensor(common_w1, top_x, input_dtype)

            if lora_name in self.w3_.loras_:
                w3 = self.w3_.loras_[lora_name].forward(
                    _mixtral_slice_tensor(common_w3, top_x, input_dtype),
                    _mixtral_slice_tensor(hidden_states, top_x, input_dtype, lora_data),
                )
            else:
                w3 = _mixtral_slice_tensor(common_w3, top_x, input_dtype)

            act_result = act_fn(w1) * w3
            if lora_name in self.w2_.loras_:
                final_expert_states.append(
                    self.w2_.loras_[lora_name].forward(
                        self.w2_.base_layer_.forward(act_result), act_result
                    )
                )
            else:
                final_expert_states.append(self.w2_.base_layer_.forward(act_result))

        return final_expert_states


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
    def __init__(self, layer_id: int) -> None:
        super().__init__()
        self.layer_id_: int = layer_id
        self.self_attn_: LlamaAttention = None
        self.mlp_: FeedForward = None
        self.input_layernorm_: LlamaRMSNorm = None
        self.post_attention_layernorm_: LlamaRMSNorm = None

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

        residual = hidden_states
        hidden_states = self.input_layernorm_(hidden_states)
        # Self Attention
        hidden_states = self.self_attn_.forward(
            hidden_states,
            input_args,
            rotary_emb,
            attention_mask,
            cache_position,
            past_key_value,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm_(hidden_states)
        hidden_states, router_logits = self.mlp_.forward(hidden_states, input_args)
        hidden_states = residual + hidden_states

        return hidden_states, *router_logits


class LlamaEmbedding(nn.Module):
    def __init__(self, embedding: torch.Tensor, pad_token: int):
        super().__init__()
        self.token_embedding_: torch.Tensor = embedding
        self.padding_idx_: int = pad_token

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        data = F.embedding(tokens, self.token_embedding_, padding_idx=self.padding_idx_)
        return data


class LlamaForCausalLM(LLMForCausalLM):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config_ = config
        self.padding_idx_ = config.pad_token_id_
        self.vocab_size_ = config.vocab_size_
        self.embed_tokens_: LlamaEmbedding = None
        self.norm_: LlamaRMSNorm = None
        self.rotary_emb_ = LlamaRotaryEmbedding(config)
        self.lm_head_ = nn.Linear(
            config.dim_,
            config.vocab_size_,
            bias=False,
            dtype=config.dtype_,
            device=config.device_,
        )
        self.layers_: List[LlamaDecoderLayer] = []

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

    def model_config(self) -> LlamaConfig:
        return self.config_

    @staticmethod
    def from_pretrained(
        llm_model: modeling_llama.LlamaForCausalLM,
        attn_impl: str = "eager",
        use_sliding_window: bool = False,
        device: str = backend.default_device_name(),
    ):
        assert not use_sliding_window, "Llama model does not support SWA."
        llm_config: modeling_llama.LlamaConfig = llm_model.config
        llm_args = LlamaConfig(
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
            rope_scaling_=llm_config.rope_scaling,
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
            llm_model.model.embed_tokens.weight, llm_args.pad_token_id_
        )
        model.norm_ = LlamaRMSNorm(llm_model.model.norm.weight, llm_args.rms_norm_eps_)
        copy_parameters(llm_model.lm_head, model.lm_head_)

        for idx, layer in enumerate(llm_model.model.layers):
            decoder = LlamaDecoderLayer(idx)
            decoder.self_attn_ = LLAMA_ATTENTION_CLASSES[llm_args.attn_implementation_](
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
