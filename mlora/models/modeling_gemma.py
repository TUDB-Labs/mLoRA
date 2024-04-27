from mlora.common import (
    CheckpointRecomputeFunction as CheckpointFunction,
    FeedForward,
)
from mlora.models.modeling_llama import (
    LlamaConfig,
    LLAMA_ATTENTION_CLASSES as GEMMA_ATTENTION_CLASSES,
    LlamaMLP,
    LlamaDecoderLayer,
    LlamaForCausalLM,
)
from mlora.backends import get_backend
from mlora.utils import copy_parameters

from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers.models.gemma.modeling_gemma as modeling_gemma


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
        data = F.embedding(tokens, self.token_embedding_,
                           padding_idx=self.padding_idx_)
        # normalized
        # Gemma downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.normalizer_, dtype=data.dtype)
        return data * normalizer


class GemmaSequentialWrapper(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.wrapper_module_ = module

    def name(self) -> str:
        return type(self.wrapper_module_).__name__

    def forward(self, input: Tuple) -> Tuple:
        module_name = self.name()

        if module_name == "GemmaEmbedding":
            output = self.wrapper_module_.forward(input[0])
            if input[-1]:
                output = output.requires_grad_(True)
            return (output, ) + input[1:]
        elif module_name == "GemmaRMSNorm":
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


class GemmaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__(config)

    def sequential_module(self) -> OrderedDict:
        seq_module = OrderedDict()

        seq_module.update(
            {"embedding": GemmaSequentialWrapper(self.embed_tokens_)})
        seq_module.move_to_end("embedding")

        for index, layer in enumerate(self.layers_):
            layer_name = f"layer{index}"
            seq_module.update({layer_name: GemmaSequentialWrapper(layer)})
            seq_module.move_to_end(layer_name)

        seq_module.update(
            {"norm": GemmaSequentialWrapper(self.norm_)})
        seq_module.move_to_end("norm")

        return seq_module

    @staticmethod
    def from_pretrained(llm_model: modeling_gemma.GemmaForCausalLM,
                        attn_impl: str = "eager",
                        use_sliding_window: bool = False,
                        device: str = get_backend().device_name() + ":0"):
        assert not use_sliding_window, "Gemma model does not support SWA."
        llm_config: modeling_gemma.GemmaConfig = llm_model.config
        llm_args = LlamaConfig(
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

        model = GemmaForCausalLM(llm_args)
        llm_model.requires_grad_(False)
        model.embed_tokens_ = GemmaEmbedding(
            llm_model.model.embed_tokens.weight, llm_args.pad_token_id_, llm_args.dim_**0.5)
        model.norm_ = GemmaRMSNorm(
            llm_model.model.norm.weight, llm_args.rms_norm_eps_)
        copy_parameters(llm_model.lm_head, model.lm_head_)

        for idx, layer in enumerate(llm_model.model.layers):
            decoder = LlamaDecoderLayer()
            decoder.layer_id_ = idx
            decoder.self_attn_ = GEMMA_ATTENTION_CLASSES[llm_args.attn_implementation_](
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
            decoder.input_layernorm_ = GemmaRMSNorm(
                layer.input_layernorm.weight, llm_args.rms_norm_eps_)
            decoder.post_attention_layernorm_ = GemmaRMSNorm(
                layer.post_attention_layernorm.weight, llm_args.rms_norm_eps_)
            model.layers_.append(decoder)

        return model
