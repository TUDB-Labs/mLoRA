import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gemma import modeling_gemma

from mlora.backends import backend
from mlora.common import FeedForward
from mlora.models.modeling_llama import (
    LLAMA_ATTENTION_CLASSES as GEMMA_ATTENTION_CLASSES,
)
from mlora.models.modeling_llama import (
    LlamaConfig,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
)
from mlora.utils import copy_parameters


class GemmaRMSNorm(nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.norm_eps_ = eps
        self.weight_ = weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.norm_eps_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.to(torch.float32))
        # Llama does x.to(float16) * w whilst Gemma is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight_.to(torch.float32))
        return output.to(x.dtype)


class GemmaEmbedding(nn.Module):
    def __init__(self, embedding: torch.Tensor, pad_token: int, normalizer: float):
        super().__init__()
        self.token_embedding_: torch.Tensor = embedding
        self.padding_idx_: int = pad_token
        self.normalizer_: float = normalizer

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        data = F.embedding(tokens, self.token_embedding_, padding_idx=self.padding_idx_)
        # normalized
        # Gemma downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.normalizer_, dtype=data.dtype)
        return data * normalizer


def _patch_hidden_act(config: modeling_gemma.GemmaConfig) -> str:
    if hasattr(config, "hidden_activation") and config.hidden_activation is not None:
        return config.hidden_activation
    else:
        return config.hidden_act


class GemmaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)

    @staticmethod
    def from_pretrained(
        llm_model: modeling_gemma.GemmaForCausalLM,
        attn_impl: str = "eager",
        use_sliding_window: bool = False,
        device: str = backend.default_device_name(),
    ):
        assert not use_sliding_window, "Gemma model does not support SWA."
        llm_config: modeling_gemma.GemmaConfig = llm_model.config
        llm_args = LlamaConfig(
            name_or_path_=llm_config.name_or_path,
            vocab_size_=llm_config.vocab_size,
            dim_=llm_config.hidden_size,
            head_dim_=llm_config.head_dim,
            intermediate_=llm_config.intermediate_size,
            n_layers_=llm_config.num_hidden_layers,
            n_heads_=llm_config.num_attention_heads,
            n_kv_heads_=llm_config.num_key_value_heads,
            hidden_act_=_patch_hidden_act(llm_config),
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

        model = GemmaForCausalLM(llm_args)
        llm_model.requires_grad_(False)
        model.embed_tokens_ = GemmaEmbedding(
            llm_model.model.embed_tokens.weight,
            llm_args.pad_token_id_,
            llm_args.dim_**0.5,
        )
        model.norm_ = GemmaRMSNorm(llm_model.model.norm.weight, llm_args.rms_norm_eps_)
        copy_parameters(llm_model.lm_head, model.lm_head_)

        for idx, layer in enumerate(llm_model.model.layers):
            decoder = LlamaDecoderLayer(idx)
            decoder.self_attn_ = GEMMA_ATTENTION_CLASSES[llm_args.attn_implementation_](
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
            decoder.input_layernorm_ = GemmaRMSNorm(
                layer.input_layernorm.weight, llm_args.rms_norm_eps_
            )
            decoder.post_attention_layernorm_ = GemmaRMSNorm(
                layer.post_attention_layernorm.weight, llm_args.rms_norm_eps_
            )
            model.layers_.append(decoder)

        return model
