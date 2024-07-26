from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.models.phi import modeling_phi
from transformers.models.phi.modeling_phi import (
    PhiRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
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
    eager_attention_forward,
    flash_attention_forward,
    prepare_4d_causal_attention_mask,
)
from mlora.common.mix_lora import _mixtral_slice_tensor
from mlora.utils import copy_parameters


@dataclass
class PhiConfig(LLMModelConfig):
    layer_norm_eps_: float = 1e-05
    resid_pdrop_: float = 0.0
    embd_pdrop_: float = 0.0
    rotary_emb_dim_: int = 0
    qk_layernorm_: bool = False


def apply_partial_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rotary_emb_dim: int,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q_rot, q_pass = (
        xq[..., :rotary_emb_dim],
        xq[..., rotary_emb_dim:],
    )
    k_rot, k_pass = (
        xk[..., :rotary_emb_dim],
        xk[..., rotary_emb_dim:],
    )
    # [batch_size, seq_length, num_heads, head_dim // partial_rotary_factor]
    q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin, position_ids)

    # [batch_size, seq_length, num_heads, head_dim]
    xq = torch.cat((q_rot, q_pass), dim=-1)
    xk = torch.cat((k_rot, k_pass), dim=-1)

    return xq, xk


# Multi-headed attention from 'Attention Is All You Need' paper.
class PhiAttention(LLMAttention):
    def __init__(
        self,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        dense: nn.Module,
        idx: int,
        config: PhiConfig,
    ):
        super().__init__()
        # attention
        self.wq_: Linear = Linear(q_proj, config.device_)
        self.wk_: Linear = Linear(k_proj, config.device_)
        self.wv_: Linear = Linear(v_proj, config.device_)
        self.dense_: Linear = Linear(dense, config.device_)
        # config
        self.layer_idx_ = idx
        self.dim_ = config.dim_
        self.n_heads_ = config.n_heads_
        self.n_kv_heads_ = config.n_kv_heads_
        self.n_rep_ = self.n_heads_ // self.n_kv_heads_
        self.rotary_emb_dim_ = config.rotary_emb_dim_
        self.head_dim_ = config.head_dim_
        self.dtype_ = config.dtype_
        self.is_causal_ = True
        # qk norm
        self.qk_layernorm_: bool = config.qk_layernorm_
        if self.qk_layernorm_:
            self.q_layernorm_ = nn.LayerNorm(
                self.hidden_size_ // self.num_heads_,
                eps=config.norm_eps_,
                elementwise_affine=True,
            )
            self.k_layernorm_ = nn.LayerNorm(
                self.hidden_size_ // self.num_heads_,
                eps=config.norm_eps_,
                elementwise_affine=True,
            )
        else:
            self.q_layernorm_ = nn.Identity()
            self.k_layernorm_ = nn.Identity()

    def state_dict(self) -> Dict[str, Linear]:
        return {
            "q_proj": self.wq_,
            "k_proj": self.wk_,
            "v_proj": self.wv_,
            "dense": self.dense_,
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

        xq = self.q_layernorm_(xq)
        xk = self.k_layernorm_(xk)

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

        cos, sin = rotary_emb

        # partial rotary embedding
        xq, xk = apply_partial_rotary_emb(
            xq,
            xk,
            self.rotary_emb_dim_,
            cos,
            sin,
            cache_position.unsqueeze(0),
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "partial_rotation_size": self.rotary_emb_dim_,
                "cache_position": cache_position,
            }
            xk, xv = past_key_value.update(xk, xv, self.layer_idx_, cache_kwargs)

        # before dim: batch_size, n_kv_head, seq_len, head_dim
        # after dim: batch_size, n_head, seq_len, head_dim
        xk = repeat_kv(xk, self.n_rep_)
        xv = repeat_kv(xv, self.n_rep_)

        attention_score = eager_attention_forward(
            xq.to(torch.float32), xk.to(torch.float32), xv, attention_mask
        )

        attention_score = attention_score.reshape(batch_size, max_seq_len, -1)
        attention_score = self.dense_.forward(attention_score, input_args)

        return attention_score


class PhiFlashAttention2(PhiAttention):
    def __init__(
        self,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        dense: nn.Module,
        idx: int,
        args: PhiConfig,
    ):
        assert is_flash_attn_2_available(), "Flash Attention is not available"
        super().__init__(q_proj, k_proj, v_proj, dense, idx, args)

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

        xq = self.q_layernorm_(xq)
        xk = self.k_layernorm_(xk)

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

        cos, sin = rotary_emb

        # partial rotary embedding
        xq, xk = apply_partial_rotary_emb(
            xq,
            xk,
            self.rotary_emb_dim_,
            cos,
            sin,
            cache_position.unsqueeze(0),
        )

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "partial_rotation_size": self.rotary_emb_dim_,
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

        attn_output = attn_output.reshape(
            batch_size, max_seq_len, self.dim_
        ).contiguous()
        attn_output = self.dense_.forward(attn_output, input_args)

        return attn_output


PHI_ATTENTION_CLASSES = {
    "eager": PhiAttention,
    "flash_attn": PhiFlashAttention2,
}


class PhiMLP(LLMFeedForward):
    def __init__(self, fc1: nn.Module, fc2: nn.Module, args: PhiConfig) -> None:
        super().__init__()
        # feed forward
        self.fc1_: Linear = Linear(fc1, args.device_)
        self.fc2_: Linear = Linear(fc2, args.device_)
        self.act_ = ACT2FN[args.hidden_act_]

    def state_dict(self) -> Dict[str, nn.Module]:
        return {
            "fc1": self.fc1_,
            "fc2": self.fc2_,
        }

    def _batch_forward(
        self, hidden_states: torch.Tensor, input_args: LLMModelInput
    ) -> torch.Tensor:
        hidden_states = self.fc1_.forward(hidden_states, input_args)
        hidden_states = self.act_(hidden_states)
        hidden_states = self.fc2_.forward(hidden_states, input_args)
        return hidden_states

    def _lora_forward(
        self, lora_name: str, act_fn: nn.Module, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        if lora_name in self.fc1_.loras_:
            hidden_states = self.fc1_.loras_[lora_name].forward(
                self.fc1_.base_layer_.forward(hidden_states), hidden_states
            )
        else:
            hidden_states = self.fc1_.base_layer_.forward(hidden_states)

        hidden_states = act_fn(hidden_states)

        if lora_name in self.fc2_.loras_:
            hidden_states = self.fc2_.loras_[lora_name].forward(
                self.fc2_.base_layer_.forward(hidden_states), hidden_states
            )
        else:
            hidden_states = self.fc2_.base_layer_.forward(hidden_states)

        return hidden_states

    def _mixlora_forward(
        self, moe_name, act_fn, expert_mask, hidden_states, input_dtype
    ):
        common_fc1 = self.fc1_.base_layer_.forward(hidden_states.to(input_dtype)).to(
            hidden_states.dtype
        )
        final_expert_states = []
        for expert_idx in range(expert_mask.shape[0]):
            _, top_x = torch.where(expert_mask[expert_idx])

            lora_name = f"moe.{moe_name}.experts.{expert_idx}"
            if lora_name in self.fc1_.loras_:
                lora_data = _mixtral_slice_tensor(hidden_states, top_x, input_dtype)
                act_result = act_fn(
                    self.fc1_.loras_[lora_name].forward(
                        _mixtral_slice_tensor(common_fc1, top_x, input_dtype), lora_data
                    )
                )
            else:
                act_result = act_fn(
                    _mixtral_slice_tensor(common_fc1, top_x, input_dtype)
                )

            if lora_name in self.fc2_.loras_:
                final_expert_states.append(
                    self.fc2_.loras_[lora_name].forward(
                        self.fc2_.base_layer_.forward(act_result), act_result
                    )
                )
            else:
                final_expert_states.append(self.fc2_.base_layer_.forward(act_result))

        return final_expert_states


class PhiDecoderLayer(LLMDecoder):
    def __init__(
        self, layer_id: int, self_attn: LLMAttention, mlp: FeedForward, args: PhiConfig
    ) -> None:
        super().__init__()
        self.layer_id_: int = layer_id
        self.self_attn_ = self_attn
        self.mlp_ = mlp
        self.input_layernorm_ = nn.LayerNorm(
            args.dim_, eps=args.layer_norm_eps_, dtype=args.dtype_, device=args.device_
        )
        self.resid_pdrop_ = args.resid_pdrop_

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
        attn_outputs = self.self_attn_.forward(
            hidden_states,
            input_args,
            rotary_emb,
            attention_mask,
            cache_position,
            past_key_value,
        )
        attn_outputs = F.dropout(
            attn_outputs, self.resid_pdrop_, not input_args.inference_mode_
        )
        # Fully Connected
        feed_forward_outputs, router_logits = self.mlp_.forward(
            hidden_states, input_args
        )
        feed_forward_outputs = F.dropout(
            feed_forward_outputs, self.resid_pdrop_, not input_args.inference_mode_
        )
        hidden_states = attn_outputs + feed_forward_outputs + residual

        return hidden_states, *router_logits


class PhiEmbedding(nn.Module):
    def __init__(self, config: PhiConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size_,
            config.dim_,
            config.pad_token_id_,
            dtype=config.dtype_,
            device=config.device_,
        )
        self.embed_dropout = nn.Dropout(config.embd_pdrop_)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        return self.embed_dropout(inputs_embeds)


class PhiLayerNorm(nn.Module):
    def __init__(self, config: PhiConfig) -> None:
        super().__init__()
        self.layernorm_ = nn.LayerNorm(
            config.dim_,
            eps=config.layer_norm_eps_,
            dtype=config.dtype_,
            device=config.device_,
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.layernorm_(data)


class PhiForCausalLM(LLMForCausalLM):
    def __init__(self, config: PhiConfig) -> None:
        super().__init__()
        self.config_ = config
        self.padding_idx_ = config.pad_token_id_
        self.vocab_size_ = config.vocab_size_
        self.embed_tokens_ = PhiEmbedding(config)
        self.final_layernorm_ = PhiLayerNorm(config)
        self.rotary_emb_ = PhiRotaryEmbedding(
            dim=config.rotary_emb_dim_,
            max_position_embeddings=config.max_seq_len_,
            base=config.rope_theta_,
            device=config.device_,
        )
        self.lm_head_ = nn.Linear(
            config.dim_,
            config.vocab_size_,
            bias=True,
            dtype=config.dtype_,
            device=config.device_,
        )
        self.layers_: List[PhiDecoderLayer] = []

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens_(input_ids)

    def rotary_embed(
        self, input_tensor: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rotary_emb_(input_tensor, seq_len=position_ids[-1, -1] + 1)

    def decoder_stack(self) -> List[LLMDecoder]:
        return self.layers_

    def norm(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.final_layernorm_(hidden_states)

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

    def model_config(self) -> PhiConfig:
        return self.config_

    @staticmethod
    def from_pretrained(
        llm_model: modeling_phi.PhiForCausalLM,
        attn_impl: str = "eager",
        use_sliding_window: bool = False,
        device: str = backend.default_device_name(),
    ):
        assert not use_sliding_window, "Phi model does not support SWA."
        llm_config: modeling_phi.PhiConfig = llm_model.config
        llm_args = PhiConfig(
            name_or_path_=llm_config.name_or_path,
            vocab_size_=llm_config.vocab_size,
            dim_=llm_config.hidden_size,
            head_dim_=llm_config.hidden_size // llm_config.num_attention_heads,
            intermediate_=llm_config.intermediate_size,
            n_layers_=llm_config.num_hidden_layers,
            n_heads_=llm_config.num_attention_heads,
            n_kv_heads_=llm_config.num_key_value_heads,
            hidden_act_=llm_config.hidden_act,
            resid_pdrop_=llm_config.resid_pdrop,
            embd_pdrop_=llm_config.embd_pdrop,
            max_seq_len_=llm_config.max_position_embeddings,
            layer_norm_eps_=llm_config.layer_norm_eps,
            rope_theta_=llm_config.rope_theta,
            partial_rotary_factor_=llm_config.partial_rotary_factor,
            qk_layernorm_=llm_config.qk_layernorm,
            pad_token_id_=llm_config.pad_token_id,
            attn_implementation_=attn_impl,
            device_=torch.device(device),
            dtype_=llm_model.dtype,
        )

        llm_args.rotary_emb_dim_ = int(
            llm_args.partial_rotary_factor_ * llm_args.head_dim_
        )

        if llm_args.pad_token_id_ is None:
            llm_args.pad_token_id_ = -1

        model = PhiForCausalLM(llm_args)
        llm_model.requires_grad_(False)
        copy_parameters(llm_model.model.embed_tokens, model.embed_tokens_.embed_tokens)
        copy_parameters(
            llm_model.model.final_layernorm, model.final_layernorm_.layernorm_
        )
        copy_parameters(llm_model.lm_head, model.lm_head_)

        for idx, layer in enumerate(llm_model.model.layers):
            decoder = PhiDecoderLayer(
                idx,
                PHI_ATTENTION_CLASSES[llm_args.attn_implementation_](
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                    layer.self_attn.dense,
                    idx,
                    llm_args,
                ),
                FeedForward(
                    PhiMLP(
                        layer.mlp.fc1,
                        layer.mlp.fc2,
                        llm_args,
                    )
                ),
                llm_args,
            )
            copy_parameters(layer.input_layernorm, decoder.input_layernorm_)
            model.layers_.append(decoder)

        return model
