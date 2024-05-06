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
    CHECKPOINT_CLASSES,
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
import transformers.models.phi.modeling_phi as modeling_phi

if _flash_attn_available:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input


@dataclass
class PhiConfig(LLMModelArgs):
    partial_rotary_factor_: float = 0.5
    layer_norm_eps_: float = 1e-05
    resid_pdrop_: float = 0.0
    embd_pdrop_: float = 0.0
    qk_layernorm_: bool = False


@torch.jit.script
def apply_partial_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, rotary_emb_dim: int, seq_len: int,
                             cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    q_rot, q_pass = (
        xq[..., : rotary_emb_dim],
        xq[..., rotary_emb_dim:],
    )
    k_rot, k_pass = (
        xk[..., : rotary_emb_dim],
        xk[..., rotary_emb_dim:],
    )
    # [batch_size, seq_length, num_heads, head_dim // partial_rotary_factor]
    q_rot, k_rot = apply_rotary_emb(q_rot, k_rot, seq_len, cos, sin)

    # [batch_size, seq_length, num_heads, head_dim]
    xq = torch.cat((q_rot, q_pass), dim=-1)
    xk = torch.cat((k_rot, k_pass), dim=-1)

    return xq, xk


# Multi-headed attention from 'Attention Is All You Need' paper.
class PhiAttention(LLMAttention):
    def __init__(self, q_proj: nn.Module, k_proj: nn.Module, v_proj: nn.Module, dense: nn.Module, args: PhiConfig):
        super().__init__()
        # attention
        self.wq_: Linear = Linear(q_proj, args.device_)
        self.wk_: Linear = Linear(k_proj, args.device_)
        self.wv_: Linear = Linear(v_proj, args.device_)
        self.dense_: Linear = Linear(dense, args.device_)
        # config
        self.dim_ = args.dim_
        self.n_heads_ = args.n_heads_
        self.n_kv_heads_ = args.n_kv_heads_
        self.n_rep_ = self.n_heads_ // self.n_kv_heads_
        self.head_dim_ = args.dim_ // args.n_heads_
        self.dtype_ = args.dtype_
        self.is_causal_ = True
        # cos and sin
        self.rotary_emb_dim_ = int(
            args.partial_rotary_factor_ * self.head_dim_)
        self.cos_, self.sin_ = precompute_rope_angle(
            self.rotary_emb_dim_, args.max_seq_len_, args.rope_theta_, args.device_)
        # qk norm
        self.qk_layernorm_: bool = args.qk_layernorm_
        if self.qk_layernorm_:
            self.q_layernorm_ = nn.LayerNorm(
                self.hidden_size_ // self.num_heads_, eps=args.norm_eps_, elementwise_affine=True
            )
            self.k_layernorm_ = nn.LayerNorm(
                self.hidden_size_ // self.num_heads_, eps=args.norm_eps_, elementwise_affine=True
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

    def forward(self,
                hidden_states: torch.Tensor,
                input_args: MultiLoraBatchData,
                attention_mask: Optional[torch.Tensor] = None):
        batch_size, max_seq_len, _ = hidden_states.shape

        xq = self.wq_.forward(hidden_states, input_args)
        xk = self.wk_.forward(hidden_states, input_args)
        xv = self.wv_.forward(hidden_states, input_args)

        xq = self.q_layernorm_(xq)
        xk = self.k_layernorm_(xk)

        # conver shape to multi head
        xq = xq.view(batch_size, max_seq_len, self.n_heads_,
                     self.head_dim_).transpose(1, 2)
        xk = xk.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)
        xv = xv.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)

        # partial rotary embedding
        assert xq.dtype == xk.dtype
        xq, xk = apply_partial_rotary_emb(
            xq, xk, self.rotary_emb_dim_, max_seq_len, self.cos_, self.sin_)

        # before dim: batch_size, n_kv_head, seq_len, head_dim
        # after dim: batch_size, n_head, seq_len, head_dim
        xk = repeat_kv(xk, self.n_rep_)
        xv = repeat_kv(xv, self.n_rep_)

        attention_score = scaled_dot_product_attention(
            xq.to(torch.float32), xk.to(torch.float32), xv, attention_mask)

        attention_score = attention_score.reshape(batch_size, max_seq_len, -1)
        attention_score = self.dense_.forward(attention_score, input_args)

        return attention_score


class PhiXformersAttention(PhiAttention):
    def __init__(self, q_proj: nn.Module, k_proj: nn.Module, v_proj: nn.Module, dense: nn.Module, args: PhiConfig):
        assert _xformers_available, "xFormers Attention is not available"
        super().__init__(q_proj, k_proj, v_proj, dense, args)

    def forward(self,
                hidden_states: torch.Tensor,
                input_args: MultiLoraBatchData,
                attention_mask: Optional[torch.Tensor] = None):
        batch_size, max_seq_len, _ = hidden_states.shape

        xq = self.wq_.forward(hidden_states, input_args)
        xk = self.wk_.forward(hidden_states, input_args)
        xv = self.wv_.forward(hidden_states, input_args)

        xq = self.q_layernorm_(xq)
        xk = self.k_layernorm_(xk)

        # conver shape to multi head
        xq = xq.view(batch_size, max_seq_len, self.n_heads_,
                     self.head_dim_).transpose(1, 2)
        xk = xk.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)
        xv = xv.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)

        # partial rotary embedding
        assert xq.dtype == xk.dtype
        xq, xk = apply_partial_rotary_emb(
            xq, xk, self.rotary_emb_dim_, max_seq_len, self.cos_, self.sin_)

        # before dim: batch_size, n_kv_head, seq_len, head_dim
        # after dim: batch_size, n_head, seq_len, head_dim
        xk = repeat_kv(xk, self.n_rep_)
        xv = repeat_kv(xv, self.n_rep_)

        attention_score = xformers_attention(
            xq, xk, xv, attention_mask).to(xq.dtype)

        attention_score = attention_score.reshape(batch_size, max_seq_len, -1)
        attention_score = self.dense_.forward(attention_score, input_args)

        return attention_score


class PhiFlashAttention2(PhiAttention):
    def __init__(self, q_proj: nn.Module, k_proj: nn.Module, v_proj: nn.Module, dense: nn.Module, args: PhiConfig):
        assert _flash_attn_available, "Flash Attention is not available"
        super().__init__(q_proj, k_proj, v_proj, dense, args)

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
                                    self.num_heads, head_dim), indices_k
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

        xq = self.q_layernorm_(xq)
        xk = self.k_layernorm_(xk)

        # conver shape to multi head
        xq = xq.view(batch_size, max_seq_len, self.n_heads_,
                     self.head_dim_).transpose(1, 2)
        xk = xk.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)
        xv = xv.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)

        # partial rotary embedding
        assert xq.dtype == xk.dtype
        xq, xk = apply_partial_rotary_emb(
            xq, xk, self.rotary_emb_dim_, max_seq_len, self.cos_, self.sin_)

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
        attn_output = self.dense_.forward(attn_output, input_args)

        return attn_output


PHI_ATTENTION_CLASSES = {
    "eager": PhiAttention,
    "xformers": PhiXformersAttention,
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

    def _batch_forward(self, hidden_states: torch.Tensor, input_args: MultiLoraBatchData) -> torch.Tensor:
        hidden_states = self.fc1_.forward(hidden_states, input_args)
        hidden_states = self.act_(hidden_states)
        hidden_states = self.fc2_.forward(hidden_states, input_args)
        return hidden_states

    def _lora_forward(
            self, lora_name: str, act_fn: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        if lora_name in self.fc1_.loras_:
            hidden_states = self.fc1_.loras_[lora_name].forward(
                self.fc1_.base_layer_.forward(hidden_states), hidden_states)
        else:
            hidden_states = self.fc1_.base_layer_.forward(hidden_states)

        hidden_states = act_fn(hidden_states)

        if lora_name in self.fc2_.loras_:
            hidden_states = self.fc2_.loras_[lora_name].forward(
                self.fc2_.base_layer_.forward(hidden_states), hidden_states)
        else:
            hidden_states = self.fc2_.base_layer_.forward(hidden_states)

        return hidden_states


class PhiDecoderLayer(LLMDecoder):
    def __init__(self, layer_id: int, self_attn: LLMAttention, mlp: FeedForward, args: PhiConfig) -> None:
        super().__init__()
        self.layer_id_: int = layer_id
        self.self_attn_ = self_attn
        self.mlp_ = mlp
        self.input_layernorm_ = nn.LayerNorm(
            args.dim_, eps=args.layer_norm_eps_, dtype=args.dtype_, device=args.device_)
        self.resid_pdrop_ = args.resid_pdrop_

    def state_dict(self) -> Dict[str, nn.Module]:
        linear_layers = self.self_attn_.state_dict()
        linear_layers.update(self.mlp_.state_dict())
        return linear_layers

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                input_args: MultiLoraBatchData):
        residual = hidden_states
        hidden_states = self.input_layernorm_(hidden_states)
        # Self Attention
        attn_outputs = self.self_attn_.forward(
            hidden_states, input_args, attention_mask)
        attn_outputs = F.dropout(
            attn_outputs, self.resid_pdrop_, not input_args.inference_mode_)
        # Fully Connected
        feed_forward_outputs, router_logits = self.mlp_.forward(
            hidden_states, input_args)
        feed_forward_outputs = F.dropout(
            feed_forward_outputs, self.resid_pdrop_, not input_args.inference_mode_)
        hidden_states = attn_outputs + feed_forward_outputs + residual

        return hidden_states, *router_logits


class PhiEmbedding(nn.Module):
    def __init__(self, config: PhiConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size_, config.dim_, config.pad_token_id_,
                                         dtype=config.dtype_, device=config.device_)
        self.embed_dropout = nn.Dropout(config.embd_pdrop_)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        inputs_embeds = self.embed_tokens(input_ids)
        return self.embed_dropout(inputs_embeds)


class PhiLayerNorm(nn.Module):
    def __init__(self, config: PhiConfig) -> None:
        super().__init__()
        self.layernorm_ = nn.LayerNorm(config.dim_, eps=config.layer_norm_eps_,
                                       dtype=config.dtype_, device=config.device_)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.layernorm_(data)


class PhiSequentialWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.wrapper_module_ = module

    def name(self) -> str:
        return type(self.wrapper_module_).__name__

    def forward(self, input: Tuple) -> Tuple:
        module_name = self.name()

        if module_name == "PhiEmbedding":
            output = self.wrapper_module_.forward(input[0])
            if input[-1].gradient_checkpoint_ != "none":
                output = output.requires_grad_(True)
            return (output,) + input[1:]
        elif module_name == "PhiLayerNorm":
            output = self.wrapper_module_.forward(input[0])
            return (output,) + input[1:]
        elif module_name == "PhiDecoderLayer":
            outputs = CHECKPOINT_CLASSES[input[-1].gradient_checkpoint_](
                self.wrapper_module_.forward, *input)
            if len(outputs) > 1:
                self.router_probs_ = outputs[1:]
            return (outputs[0],) + input[1:]
        else:
            raise f"module invalid: {module_name}"


class PhiForCausalLM(LLMForCausalLM):
    def __init__(self, config: PhiConfig) -> None:
        self.config_ = config
        self.padding_idx_ = config.pad_token_id_
        self.vocab_size_ = config.vocab_size_
        self.embed_tokens_ = PhiEmbedding(config)
        self.final_layernorm_ = PhiLayerNorm(config)
        self.lm_head_ = nn.Linear(config.dim_, config.vocab_size_, bias=True,
                                  dtype=config.dtype_, device=config.device_)
        self.layers_: List[PhiDecoderLayer] = []

    def decoder_stack(self) -> List[LLMDecoder]:
        return self.layers_

    def sequential_module(self) -> OrderedDict:
        seq_module = OrderedDict()

        seq_module.update(
            {"embedding": PhiSequentialWrapper(self.embed_tokens_)})
        seq_module.move_to_end("embedding")

        for index, layer in enumerate(self.layers_):
            layer_name = f"layer{index}"
            seq_module.update({layer_name: PhiSequentialWrapper(layer)})
            seq_module.move_to_end(layer_name)

        seq_module.update(
            {"norm": PhiSequentialWrapper(self.final_layernorm_)})
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
    def from_pretrained(llm_model: modeling_phi.PhiForCausalLM,
                        attn_impl: str = "eager",
                        use_sliding_window: bool = False,
                        device: str = get_backend().device_name() + ":0"):
        assert not use_sliding_window, "Phi model does not support SWA."
        llm_config: modeling_phi.PhiConfig = llm_model.config
        llm_args = PhiConfig(
            name_or_path_=llm_config.name_or_path,
            vocab_size_=llm_config.vocab_size,
            dim_=llm_config.hidden_size,
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

        if llm_args.pad_token_id_ is None:
            llm_args.pad_token_id_ = -1

        model = PhiForCausalLM(llm_args)
        llm_model.requires_grad_(False)
        copy_parameters(llm_model.model.embed_tokens,
                        model.embed_tokens_.embed_tokens)
        copy_parameters(llm_model.model.final_layernorm,
                        model.final_layernorm_.layernorm_)
        copy_parameters(llm_model.lm_head,
                        model.lm_head_)

        for idx, layer in enumerate(llm_model.model.layers):
            decoder = PhiDecoderLayer(
                idx,
                PHI_ATTENTION_CLASSES[llm_args.attn_implementation_](
                    layer.self_attn.q_proj,
                    layer.self_attn.k_proj,
                    layer.self_attn.v_proj,
                    layer.self_attn.dense,
                    llm_args,
                ),
                FeedForward(PhiMLP(
                    layer.mlp.fc1,
                    layer.mlp.fc2,
                    llm_args,
                )),
                llm_args,
            )
            copy_parameters(layer.input_layernorm, decoder.input_layernorm_)
            model.layers_.append(decoder)

        return model
