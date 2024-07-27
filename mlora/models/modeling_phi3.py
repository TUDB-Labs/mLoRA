import math
from mlora.common import (
    CHECKPOINT_CLASSES,
    Cache,
    FeedForward,
    Linear,
    LLMAttention,
    LLMDecoder,
    LLMFeedForward,
    LLMForCausalLM,
    LLMModelConfig,
    LLMModelInput,
    Masks,
    eager_attention_forward,
    flash_attention_forward,
    prepare_4d_causal_attention_mask,
)
from mlora.models.modeling_llama import (
    LlamaConfig,
    LLAMA_ATTENTION_CLASSES as PHI3_ATTENTION_CLASSES,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaForCausalLM,
)
from mlora.common.mix_lora import _mixtral_slice_tensor
from mlora.backends import backend
from mlora.utils import copy_parameters, is_package_available

from typing import Tuple, Dict, List, Optional
from transformers.activations import ACT2FN
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    replace_return_docstrings,
)
from collections import OrderedDict
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.phi3.modeling_phi3 import apply_rotary_pos_emb, repeat_kv
from transformers.utils import is_flash_attn_2_available
from .modeling_gemma2 import Gemma2RotaryEmbedding

from transformers.models.phi3.modeling_phi3 import(
    Phi3RotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv
)
from transformers.utils import is_flash_attn_2_available

if is_flash_attn_2_available:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input


@dataclass
class Phi3Config(LLMModelConfig):
    rms_norm_eps_: float = 1e-6
    max_position_embeddings: int = 4096
    original_max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rope_scaling: dict = None
    use_sliding_window_: bool = False
    sliding_window_: int = 4096
    resid_pdrop: float = 0.0
    rms_norm_eps=1e-5

    

class Phi3RMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.norm_eps_ = eps
        self.weight_ = weight

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        input_dtype = data.dtype
        v = data.to(torch.float32).pow(2).mean(-1, keepdim=True)
        data = data * torch.rsqrt(v + self.norm_eps_)

        print("Phi3 RMSNorm passed.")
        return (self.weight_ * data).to(input_dtype)


class Phi3Embedding(nn.Module):
    def __init__(self, embedding: torch.Tensor, pad_token: int):
        super().__init__()
        self.token_embedding_: torch.Tensor = embedding
        self.padding_idx_: int = pad_token

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        data = F.embedding(tokens, self.token_embedding_,
                           padding_idx=self.padding_idx_)
        # normalizer = torch.tensor(self.normalizer_, dtype=data.dtype)
        print("Phi3Embedding passed.")
        return data



class Phi3Attention(LLMAttention):
    def __init__(
            self, 
            qkv_proj: nn.Module,
            o_proj: nn.Module,
            layer_idx: int,
            args: Phi3Config
    ) -> None:
        super().__init__()
        # attention
        self.qkv_proj_ = Linear(qkv_proj, args.device_)
        self.o_proj_ = Linear(o_proj, args.device_)
        # config
        self.layer_idx = layer_idx
        self.args_ = args
        self.dim_ = args.dim_
        self.n_heads_ = args.n_heads_
        self.n_kv_heads_ = args.n_kv_heads_
        self.n_rep_ = self.n_heads_ // self.n_kv_heads_
        self.max_position_embeddings = args.max_position_embeddings
        self.original_max_embeddings = args.original_max_position_embeddings
        self.rope_theta_ = args.rope_theta
        self.head_dim_ = self.dim_ // self.n_heads_
        # self.head_dim = args.head_dim_
        self.dtype_ = args.dtype_
        self.is_causal_ = True
        # self.scaling_
        self.sliding_window_ = (
            args.sliding_window_
            if args.use_sliding_window_ and not bool(layer_idx % 2)
            else None
        )

    def state_dict(self) -> Dict[str, Linear]:
        return {
            "qkv_proj": self.qkv_proj_,
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

        qkv = self.qkv_proj_.forward(hidden_states, input_args)
        query_pos = self.n_heads_  * self.head_dim_
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos: query_pos + self.n_kv_heads_ * self.head_dim_]
        value_states = qkv[..., query_pos + self.n_kv_heads_ * self.head_dim_ :]

        query_states = query_states.view(bsz, q_len, self.n_heads_, self.head_dim_).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.n_kv_heads_, self.head_dim_).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.n_kv_heads_, self.head_dim_).transpose(1, 2)

        # apply rotary embedding
        cos, sin = rotary_emb  # (value_states, cache_position.unsqueeze(0))
        assert query_states.dtype == key_states.dtype
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

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


        value_states = repeat_kv(value_states, self.n_rep_)
        key_states = repeat_kv(key_states, self.n_rep_)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim_)
        
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights += causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
    
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        print("Phi3 Attention passed.")

        return self.o_proj_(attn_output, input_args)  # [8,139,3072]


class Phi3FlashAttention2(Phi3Attention):
    """
    Phi-3 flash attention module. This module inherits from `Phi3Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(
            self,
            qkv_proj: nn.Module,
            o_proj: nn.Module,
            layer_idx: int,
            args: Phi3Config,) -> None:
        assert is_flash_attn_2_available(), "Flash Attention is not available"
        super().__init__(qkv_proj, o_proj, args)
        # self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_args: LLMModelInput,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
    ):

        bsz, q_len, _ = hidden_states.size()

        # cutting
        qkv = self.qkv_proj_.forward(hidden_states, input_args)
        query_pos = self.n_heads_ * self.head_dim_
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.n_kv_heads_ * self.head_dim_]
        value_states = qkv[..., query_pos + self.n_kv_heads_ * self.head_dim_ :]

        # sin & cos
        cos, sin = self.rotary_emb_(value_states, cache_position.unsqueeze(0))  # 维度增加
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} ")

            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # Sliding windows
        # use_sliding_windows = (
        #     _flash_supports_window_size
        #     and getattr(self.config, "sliding_window", None) is not None
        #     and kv_seq_len > self.config.sliding_window
        # )

        # Activate kv_cache
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
            if _backend.is_bf16_supported():
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


        # dimosion transposing
        query_states = query_states.view(bsz, q_len, self.n_heads_, self.head_dim_).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.n_kv_heads_, self.head_dim_).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.n_kv_heads_, self.head_dim_).transpose(1, 2)

        # kv_seq_len = key_states.shape[-2] This part is about sliding windows

        # apply rotary embedding
        assert key_states.dtype == query_states.dtype
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, q_len, self.cos_, self.sin_)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if _backend.is_bf16_supported():
                target_dtype = torch.bfloat16
            else:
                target_dtype = torch.float16
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)


        # repeat k/v heads if n_kv_heads < n_heads
        # key_states = repeat_kv(key_states, self.n_rep_)
        # value_states = repeat_kv(value_states, self.n_rep_)

        # attn_dropout = self.attention_dropout if self.training else 0.0

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout = 0.0,
            # use_sliding_windows = use_sliding_windows
        ).to(input_dtype)

        # attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output

    # Copied from transformers.models.mistral.modeling_mistral.MistralFlashAttention2._flash_attention_forward
    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        # use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.
        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        use_sliding_windows = False  # 支持后删除

        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal_
        else:
            causal = self.is_causal_ and query_length != 1

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
                    # window_size=(self.config.sliding_window, self.config.sliding_window),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
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
                    # window_size=(self.config.sliding_window, self.config.sliding_window),
                )

        return attn_output

    # Copied from transformers.models.mistral.modeling_mistral.MistralFlashAttention2._upad_input
    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, n_heads_, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, n_heads_, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, n_heads_, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, n_heads_, head_dim), indices_k
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
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )
    

PHI3_ATTENTION_CLASSES = {
    "eager": Phi3Attention,
    "flash_attention_2": Phi3FlashAttention2,
}


class Phi3DecoderLayer(LLMDecoder):
    def __init__(self, layer_id: int, config: Phi3Config) -> None:
        super().__init__()
        self.layer_id_: int = layer_id
        self.self_attn_: Phi3Attention = None
        self.mlp_: FeedForward = None
        self.input_layernorm_: Phi3RMSNorm = None
        self.post_attention_layernorm_: Phi3RMSNorm = None

        self.config_ = config
        self.is_sliding_ = not bool(layer_id % 2)
        self.post_feedforward_layernorm_: Phi3RMSNorm = None
        self.sliding_window_ = config.sliding_window_

        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)
        self.post_attention_layernorm = Phi3RMSNorm(config.dim_, eps=config.rms_norm_eps)

    def state_dict(self) -> Dict[str, nn.Module]:
        linear_layers = self.self_attn_.state_dict()
        linear_layers.update(self.mlp_.state_dict())
        return linear_layers

    def forward(self,
                hidden_states: torch.Tensor,
                input_args: LLMModelInput,
                rotary_emb: Tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None,
                cache_position: Optional[torch.Tensor] = None,
                past_key_value: Optional[Cache] = None,
                ):
        residual = hidden_states
        print("[[modeling_phi3.py][phi3DecoderLayer.forward]].residual shape:\n", residual.shape)
        hidden_states = self.input_layernorm_(hidden_states)
        # Self Attention
        print("[[modeling_phi3.py][phi3DecoderLayer.forward]].hidden_states shape:\n",hidden_states.shape)
        hidden_states = self.self_attn_.forward(
            hidden_states, 
            input_args, 
            rotary_emb,
            attention_mask,
            cache_position,
            past_key_value
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm_(hidden_states)
        hidden_states, router_logits = self.mlp_.forward(hidden_states, input_args)
        hidden_states = residual + hidden_states

        print("phi3 Decoder passed.")
        return hidden_states, *router_logits
    

class Phi3SequentialWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.wrapper_module_ = module

    def name(self) -> str:
        return type(self.wrapper_module_).__name__

    def forward(self, input: Tuple) -> Tuple:
        module_name = self.name()

        if module_name == "Phi3Embedding":
            output = self.wrapper_module_.forward(input[0])  # tokens
            if input[-1].gradient_checkpoint_!= "none":  # orig.: if input[-1]:
                output = output.requires_grad_(True)
            print("Phi3 SW Embedding passed.")
            return (output, ) + input[1:]
        elif module_name == "Phi3RMSNorm":
            output = self.wrapper_module_.forward(input[0])
            print("Phi3 SW Norm passed.")
            return (output, ) + input[1:]
        elif module_name == "Phi3DecoderLayer":
            outputs = CHECKPOINT_CLASSES[input[-1].gradient_checkpoint_](
                self.wrapper_module_.forward, *input  # 这里的hidden_state已经是[8, 64, 3072]
            )
            if len(outputs) > 1:
                self.router_probs_ = outputs[1:]
            print("Phi3 SW Decoder passed.")
            return (outputs[0],) + input[1:]
        else:
            raise f"module invalid: {module_name}"


class Phi3MLP(LLMFeedForward):
     def __init__(self, gate: nn.Module, down: nn.Module, args: LlamaConfig) -> None:
        super().__init__()
        # feed forward
        self.gate_ = Linear(gate, args.device_)
        self.down_ = Linear(down, args.device_)
        self.act_ = ACT2FN[args.hidden_act_]

     def state_dict(self) -> Dict[str, nn.Module]:
        return {
            "gate_up_proj": self.gate_,
            "down_proj": self.down_,
        }

     def _batch_forward(self, hidden_states: torch.Tensor, input_args: LLMModelInput) -> torch.Tensor:
        up_states = self.gate_.forward(hidden_states, input_args)

        gate, up_states = up_states.chunk(2, dim=-1)
        up_states = up_states * self.act_(gate)

        print("Phi3MLP passed.")
        return self.down_(up_states, input_args)

     def _lora_forward(
            self, lora_name: str, act_fn: nn.Module, data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
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


class Phi3OutputLayer(nn.Module):
    def __init__(self, config: Phi3Config):
        super().__init__()
        self.lm_head_ = nn.Linear(
            config.dim_,
            config.vocab_size_,
            bias=False,
            dtype=config.dtype_,
            device=config.device_,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head_(hidden_states)
        return logits


class Phi3ForCausalLM(LLMForCausalLM):
    #father class? transformer.PreTrainedModel?
    config_class = Phi3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Phi3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_cache_class = True
    _tied_weights_keys = ["lm_head.weight"]

    _version = "0.0.5"

    def __init__(self, config: Phi3Config) -> None:
        super().__init__()
        self.config_ = config
        self.padding_idx_ = config.pad_token_id_
        self.vocab_size_ = config.vocab_size_
        self.embed_tokens_: Phi3Embedding = None
        self.norm_: Phi3Embedding = None
        self.rotary_emb_ = Gemma2RotaryEmbedding(  # 此部分等效于 init_rope()
            # args.head_dim_,  # here should be 96 (3072/32)
            config.dim_//config.n_heads_,
            max_position_embeddings = config.max_seq_len_,
            base = config.rope_theta_,
            device = config.device_,
        )
        self.lm_head_ = nn.Linear(
            config.dim_,
            config.vocab_size_,
            bias=False,
            dtype=config.dtype_,
            device=config.device_,
        )
        self.layers_: List[Phi3DecoderLayer] = []

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

    def sequential_module(self) -> OrderedDict:
        seq_module = OrderedDict()

        seq_module.update(
            {"embedding": Phi3SequentialWrapper(self.embed_tokens_)})
        seq_module.move_to_end("embedding")

        for index, layer in enumerate(self.layers_):
            layer_name = f"layer{index}"
            seq_module.update({layer_name: Phi3SequentialWrapper(layer)})
            seq_module.move_to_end(layer_name)

        seq_module.update(
            {"norm": Phi3SequentialWrapper(self.norm_)})
        seq_module.move_to_end("norm")

        return seq_module

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

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def cache_implementation(self) -> str:
        if self.config_.use_sliding_window_ and self.config_.sliding_window_:
            return "hybrid"
        else:
            return "dynamic"

    def model_config(self) -> Phi3Config:
        return self.config_
    
    @staticmethod
    def from_pretrained(llm_model,
                        attn_impl: str = "eager",
                        use_sliding_window: bool = False,
                        device: str = backend.default_device_name()):
        assert not use_sliding_window, "Phi3 model does not support SWA."
        llm_config = llm_model.config
        llm_args = Phi3Config(
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
            device_=torch.device(device),
            dtype_=llm_model.dtype,
        )

        if llm_args.pad_token_id_ is None:
            llm_args.pad_token_id_ = -1

        model = Phi3ForCausalLM(llm_args)
        llm_model.requires_grad_(False)
        model.embed_tokens_ = Phi3Embedding(
            llm_model.model.embed_tokens.weight, llm_args.pad_token_id_
            )
        model.norm_ = Phi3RMSNorm(
            llm_model.model.norm.weight, llm_args.rms_norm_eps_)
        # copy_parameters(llm_model.model.embed_tokens, model.embed_tokens_.embed_tokens)
        # copy_parameters(llm_model.model.final_layernorm, model.final_layernorm_.layernorm_)
        copy_parameters(llm_model.lm_head, model.lm_head_)

        for idx, layer in enumerate(llm_model.model.layers):
            decoder = Phi3DecoderLayer(idx, llm_args)
            decoder.layer_id_ = idx
            decoder.self_attn_ = PHI3_ATTENTION_CLASSES[llm_args.attn_implementation_](
                layer.self_attn.qkv_proj,
                layer.self_attn.o_proj,
                idx,
                llm_args,
            )
            decoder.mlp_ = FeedForward(
                Phi3MLP(
                    layer.mlp.gate_up_proj,
                    layer.mlp.down_proj,
                    llm_args,
                )
            )
            decoder.input_layernorm_ = Phi3RMSNorm(
                layer.input_layernorm.weight, llm_args.rms_norm_eps_)
            decoder.post_attention_layernorm_ = Phi3RMSNorm(
                layer.post_attention_layernorm.weight, llm_args.rms_norm_eps_)
            model.layers_.append(decoder)

        print("phi3FormPretrained passed ")
        return model





