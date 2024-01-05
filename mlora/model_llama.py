from mlora.modelargs import LoraConfig, MixConfig, LLMModelArgs, MultiLoraBatchData, lora_config_factory
from mlora.checkpoint import CheckpointRecomputeFunction
from mlora.model import repeat_kv, apply_rotary_emb, precompute_rope_angle, precompute_mask, precompute_mask_for_inference
from mlora.model import KVCache, LLMModel, RMSNorm
from mlora.generate import GenerateConfig
from mlora.FeedForward import FeedForward
from mlora.LoraLiner import Linear

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import xformers.ops
import xformers.ops.fmha.attn_bias
from typing import List, Dict, Tuple, Optional
from huggingface_hub import snapshot_download
from transformers import LlamaForCausalLM
from collections import OrderedDict
import logging
import json
import math
import os


class Embedding(torch.nn.Module):
    def __init__(self, embedding: torch.Tensor, pad_token: int):
        super().__init__()
        self.token_embedding_: torch.Tensor = embedding
        self.padding_idx_: int = pad_token

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        data = F.embedding(tokens, self.token_embedding_,
                           padding_idx=self.padding_idx_)
        data.requires_grad_(True)
        return data


class OutputLayer(torch.nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight_: torch.Tensor = weight

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return data @ self.weight_.transpose(0, 1)


class RMSNormLayer(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.norm_eps_ = eps
        self.weight_ = weight

    def _norm(self, data: torch.Tensor) -> torch.Tensor:
        return data * torch.rsqrt(+ self.norm_eps_)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        input_dtype = data.dtype
        v = data.to(torch.float32).pow(2).mean(-1, keepdim=True)
        data = data * torch.rsqrt(v + self.norm_eps_)

        return (self.weight_ * data).to(input_dtype)


class Transformer(torch.nn.Module):
    def __init__(self, layer_id: int, args: LLMModelArgs):
        super().__init__()
        # attention
        self.attention_norm_: RMSNorm = None  # dim
        self.wq_: Linear = None  # dim * dim
        self.wk_: Linear = None  # dim * dim
        self.wv_: Linear = None  # dim * dim
        self.wo_: Linear = None  # dim * dim
        # feed forward
        self.ffn_: FeedForward = None

        # other arg
        self.layer_id_ = layer_id
        self.norm_eps_ = args.norm_eps_
        self.n_heads_ = args.n_heads_
        self.n_kv_heads_ = args.n_kv_heads_
        self.n_rep_ = self.n_heads_ // self.n_kv_heads_
        self.head_dim_ = args.dim_ // args.n_heads_

    def init_lora_layer_weight(self, config: LoraConfig, weight: Optional[Dict[str, torch.Tensor]]):
        adapter_name = config.adapter_name_
        target = config.target_modules_
        linear_layer_list = [self.wk_, self.wq_, self.wv_,
                             self.wo_, self.ffn_.w1_, self.ffn_.w2_, self.ffn_.w3_]
        linear_layer_name_list = [
            "k_proj", "q_proj", "v_proj", "o_proj", "w1_proj", "w2_proj", "w3_proj"]

        if isinstance(config, MixConfig):
            # Inject LoRA configs into FFN layer
            gate_layer_name = f"mixlora.layers.{self.layer_id_}.gate.weight"
            self.ffn_.init_moe_weight(in_features=self.n_heads_ * self.head_dim_,
                                      config=config,
                                      gate=weight if weight is None else weight[gate_layer_name])

            moe_layer_name_list = ["w1_proj", "w2_proj", "w3_proj"]
            init_moe = True
        else:
            moe_layer_name_list = []
            init_moe = False

        for idx, layer_name in enumerate(linear_layer_name_list):
            if layer_name in target and target[layer_name]:
                if init_moe and layer_name in moe_layer_name_list:
                    for expert_idx in range(config.num_experts_):
                        lora_a = None
                        lora_b = None
                        if weight is not None:
                            lora_a_name = f"mixlora.layers.{self.layer_id_}.experts.{expert_idx}.{layer_name}.lora_A.weight"
                            lora_b_name = f"mixlora.layers.{self.layer_id_}.experts.{expert_idx}.{layer_name}.lora_B.weight"
                            if lora_a_name not in weight:
                                raise f"can not found the layer {lora_a_name} in model"
                            if lora_b_name not in weight:
                                raise f"can not found the layer {lora_b_name} in model"
                            lora_a = weight[lora_a_name]
                            lora_b = weight[lora_b_name]

                        linear_layer_list[idx].init_lora_weight(
                            f"moe.{adapter_name}.experts.{expert_idx}",
                            config.lora_r_, config.lora_alpha_, config.lora_dropout_, lora_a, lora_b)
                else:
                    lora_a = None
                    lora_b = None
                    if weight is not None:
                        name_prefix = "mixlora.layers" if init_moe else "base_model.model.model.layers"
                        lora_a_name = f"{name_prefix}.{self.layer_id_}.self_attn.{layer_name}.lora_A.weight"
                        lora_b_name = f"{name_prefix}.{self.layer_id_}.self_attn.{layer_name}.lora_B.weight"

                        if lora_a_name not in weight:
                            raise f"can not found the layer {lora_a_name} in model"
                        if lora_b_name not in weight:
                            raise f"can not found the layer {lora_b_name} in model"
                        lora_a = weight[lora_a_name]
                        lora_b = weight[lora_b_name]

                    linear_layer_list[idx].init_lora_weight(
                        adapter_name, config.lora_r_, config.lora_alpha_, config.lora_dropout_, lora_a, lora_b)

    # @torch.compile
    def forward(self,
                data: torch.Tensor,
                mask: torch.Tensor,
                rope_angle: Tuple[torch.Tensor, torch.Tensor],
                input_args: MultiLoraBatchData,
                router_logits: Tuple[List] = None,
                kv_cache: KVCache = None):
        batch_size, max_seq_len, _ = data.shape

        attention_norm_data = self.attention_norm_.forward(data)

        xq = self.wq_.forward(attention_norm_data, input_args)
        xk = self.wk_.forward(attention_norm_data, input_args)
        xv = self.wv_.forward(attention_norm_data, input_args)

        # conver shape to multi head
        xq = xq.view(batch_size, max_seq_len, self.n_heads_, self.head_dim_)
        xk = xk.view(batch_size, max_seq_len, self.n_kv_heads_, self.head_dim_)
        xv = xv.view(batch_size, max_seq_len, self.n_kv_heads_, self.head_dim_)

        # apply rotary embedding
        xq, xk = apply_rotary_emb(xq, xk, rope_angle)

        # apply kv cache
        if kv_cache is not None:
            xk, xv = kv_cache.update(
                xk, xv, self.layer_id_, batch_size, max_seq_len)

        # for llama2 need to repeat the heads
        # before dim: batch_size, seq_len, n_kv_head, head_dim
        # after dim: batch_size, seq_len, n_head, head_dim
        xk = repeat_kv(xk, self.n_rep_)
        xv = repeat_kv(xv, self.n_rep_)

        if kv_cache is not None:
            xq = xq.transpose(1, 2)
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)
            attention_score = torch.matmul(xq, xk.transpose(2, 3)
                                           ) / math.sqrt(self.head_dim_)
            if mask is not None:
                attention_score = attention_score + mask
            attention_score = F.softmax(
                attention_score.float(), dim=-1).type_as(xq)
            attention_score = torch.matmul(attention_score, xv)
            attention_score = attention_score.transpose(1, 2).contiguous()
        else:
            # use xformers when training
            attention_score = xformers.ops.memory_efficient_attention(
                xq, xk, xv, mask)

        attention_score = attention_score.view(batch_size, max_seq_len, -1)

        # get output attention score
        data = data + self.wo_.forward(attention_score, input_args)

        # feed forward fully connected
        data = data + self.ffn_.forward(data, router_logits, input_args)

        return data


class LlamaSequentialWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.wrapper_module_ = module

    def name(self) -> str:
        return type(self.wrapper_module_).__name__

    def forward(self, input: Tuple) -> Tuple:
        module_name = self.name()

        if module_name == "Embedding" or module_name == "RMSNormLayer" or module_name == "OutputLayer":
            output = self.wrapper_module_.forward(input[0])
            return (output, ) + input[1:]
        elif module_name == "Transformer":
            if input[-1]:
                output = CheckpointRecomputeFunction.apply(
                    self.wrapper_module_.forward, *input[:-1])
            else:
                output = self.wrapper_module_.forward(*input[:-1])
            return (output, ) + input[1:]
        else:
            raise f"module invalid: {module_name}"


class LlamaModel(LLMModel):
    def __init__(self, args: LLMModelArgs):
        # weight
        self.token_embedding_: Embedding = None

        self.layers_: List[Transformer] = []
        for layer_id in range(args.n_layers_):
            self.layers_.append(Transformer(layer_id, args))

        self.norm_: RMSNormLayer = None    # dim
        self.output_: OutputLayer = None   # vocab size * dim

        # cos and sin
        self.rope_angle_: Tuple[torch.Tensor, torch.Tensor] = precompute_rope_angle(
            args.dim_ // args.n_heads_, args.max_seq_len_, args.device)

        self.norm_eps_ = args.norm_eps_

        self.device_ = args.device
        self.n_heads_ = args.n_heads_
        self.vocab_size_ = args.vocab_size_
        self.pad_token_id_ = args.pad_token_id_
        self.max_seq_len_ = args.max_seq_len_
        self.dim_ = args.dim_

        # adapter type
        self.adapter_configs_: Dict[str, LoraConfig] = {}

    # train model or inference model: output is probs
    def forward(self, input: MultiLoraBatchData,
                output_router_logits: bool = False,
                kv_cache: KVCache = None) -> torch.Tensor:
        if isinstance(input.batch_tokens_, torch.Tensor):
            tokens = input.batch_tokens_.to(self.device_)
        else:
            tokens = torch.tensor(input.batch_tokens_,
                                  dtype=torch.int64).to(self.device_)

        seq_module = self.sequential_module()

        if input.inference_model_:
            assert kv_cache is not None
            if input.batch_seq_len_ > 1:
                # prepare mask
                mask = precompute_mask_for_inference(
                    input, kv_cache.seq_pos, self.device_)
            else:
                mask = None
            data = (tokens, mask, self.rope_angle_,
                    input, None, kv_cache, False)
            router_logits = None
        else:
            # prepare mask
            mask = precompute_mask(input, self.n_heads_, self.device_)
            # store routing data when training
            router_logits: Tuple[List] = tuple([] for _ in range(
                len(input.lora_batch_data_config_))) if output_router_logits else None
            data = (tokens, mask, self.rope_angle_,
                    input, router_logits, None, True)

        for seq_layer in seq_module:
            data = seq_layer.forward(data)

        return data[0], router_logits

    def init_lora_layer_weight(self, config: LoraConfig, weight: Optional[Dict[str, torch.Tensor]]):
        self.adapter_configs_[config.adapter_name_] = config
        for transformer_layer in self.layers_:
            transformer_layer.init_lora_layer_weight(config, weight)

    def from_pretrained(path: str,
                        device: str,
                        bits: int = None,
                        fp16: bool = True,
                        bf16: bool = True,
                        double_quant: bool = True,
                        quant_type: str = 'nf4',
                        ) -> LLMModel:
        if bits in [4, 8]:
            logging.info(f"Loading model with quantization, bits = {bits}.")
            from transformers import BitsAndBytesConfig
            compute_dtype = (torch.float16 if fp16 else (
                torch.bfloat16 if bf16 else torch.float32))
            llama_model = LlamaForCausalLM.from_pretrained(
                path,
                load_in_4bit=bits == 4,
                load_in_8bit=bits == 8,
                device_map=device,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=bits == 4,
                    load_in_8bit=bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=double_quant,
                    bnb_4bit_quant_type=quant_type,
                ),
                torch_dtype=(torch.float32 if fp16 else (torch.bfloat16 if bf16 else torch.float32)))
        else:
            llama_model = LlamaForCausalLM.from_pretrained(
                path,
                device_map=device,
                torch_dtype=torch.float32)

        llama_args = LLMModelArgs()
        llama_args.dim_ = llama_model.config.hidden_size
        llama_args.n_heads_ = llama_model.config.num_attention_heads
        llama_args.n_kv_heads_ = llama_args.n_heads_ if not hasattr(
            llama_model.config, "num_key_value_heads") else llama_model.config.num_key_value_heads
        llama_args.n_layers_ = llama_model.config.num_hidden_layers
        llama_args.norm_eps_ = llama_model.config.rms_norm_eps
        llama_args.vocab_size_ = llama_model.config.vocab_size
        llama_args.max_seq_len_ = 4096 if not hasattr(
            llama_model.config, "max_sequence_length") else llama_model.config.max_sequence_length
        llama_args.pad_token_id_ = llama_model.config.pad_token_id
        if llama_args.pad_token_id_ is None:
            llama_args.pad_token_id_ = -1

        llama_args.device = device

        model = LlamaModel(llama_args)

        embedding_weight = llama_model.model.embed_tokens.weight.to(
            device=device).requires_grad_(False)
        model.token_embedding_ = Embedding(
            embedding_weight, llama_args.pad_token_id_)

        output_weight = llama_model.lm_head.weight.to(
            dtype=torch.float32, device=device).requires_grad_(False)
        model.output_ = OutputLayer(output_weight)

        norm_weight = llama_model.model.norm.weight.to(
            device=device).requires_grad_(False)
        model.norm_ = RMSNormLayer(norm_weight, model.norm_eps_)

        for idx, layer in enumerate(llama_model.model.layers):
            model.layers_[idx].wq_ = Linear(
                layer.self_attn.q_proj, device=device)
            model.layers_[idx].wk_ = Linear(
                layer.self_attn.k_proj, device=device)
            model.layers_[idx].wv_ = Linear(
                layer.self_attn.v_proj, device=device)
            model.layers_[idx].wo_ = Linear(
                layer.self_attn.o_proj, device=device)
            model.layers_[idx].ffn_ = FeedForward(
                norm=RMSNorm(layer.post_attention_layernorm.weight.to(
                    device=device).requires_grad_(False), model.norm_eps_),
                w1=Linear(layer.mlp.gate_proj, device=device),
                w2=Linear(layer.mlp.down_proj, device=device),
                w3=Linear(layer.mlp.up_proj, device=device),
                device=device
            )
            model.layers_[idx].attention_norm_ = RMSNorm(
                layer.input_layernorm.weight.to(device=device).requires_grad_(False), model.norm_eps_)

        return model

    def get_train_paramas(self) -> Dict[str, List[torch.Tensor]]:
        train_paramas = {}

        for transformer_layer in self.layers_:
            for adapter_name, lora_config in self.adapter_configs_.items():
                if adapter_name not in train_paramas:
                    train_paramas[adapter_name] = []

                if adapter_name in transformer_layer.ffn_.moes_:
                    train_paramas[adapter_name].append(transformer_layer.ffn_.moes_[
                                                       adapter_name].gate_.weight)

                lora_layer_list = [transformer_layer.wq_.loras_, transformer_layer.wk_.loras_,
                                   transformer_layer.wv_.loras_, transformer_layer.wo_.loras_,
                                   transformer_layer.ffn_.w1_.loras_, transformer_layer.ffn_.w2_.loras_,
                                   transformer_layer.ffn_.w3_.loras_]

                for lora_layer in lora_layer_list:
                    if adapter_name in lora_layer:
                        train_paramas[adapter_name].append(
                            lora_layer[adapter_name].lora_a_)
                        train_paramas[adapter_name].append(
                            lora_layer[adapter_name].lora_b_)
                    elif adapter_name in transformer_layer.ffn_.moes_:
                        for expert_idx in range(lora_config.num_experts_):
                            lora_name = f"moe.{adapter_name}.experts.{expert_idx}"
                            if lora_name in lora_layer:
                                train_paramas[adapter_name].append(
                                    lora_layer[lora_name].lora_a_)
                                train_paramas[adapter_name].append(
                                    lora_layer[lora_name].lora_b_)

        return train_paramas

    def get_generate_paramas(self) -> Dict[str, GenerateConfig]:
        generate_paramas = {}
        for adapter_name in self.adapter_configs_.keys():
            generate_paramas[adapter_name] = GenerateConfig(
                adapter_name_=adapter_name)
        return generate_paramas

    def get_lora_weight_dict(self, lora_name: str) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        # return the lora weight and target_module's name
        lora_weight_dict = {}
        for idx, transformer_layer in enumerate(self.layers_):
            if isinstance(self.adapter_configs_[lora_name], MixConfig):
                layer_prefix_name = f"mixlora.layers.{idx}.self_attn."
            else:
                layer_prefix_name = f"base_model.model.model.layers.{idx}.self_attn."

            lora_layer_list = [transformer_layer.wq_, transformer_layer.wk_,
                               transformer_layer.wv_, transformer_layer.wo_,
                               transformer_layer.ffn_.w1_, transformer_layer.ffn_.w2_,
                               transformer_layer.ffn_.w3_]
            lora_layer_name_list = [
                "q_proj", "k_proj", "v_proj", "o_proj", "w1_proj", "w2_proj", "w3_proj"]
            for idx, lora_layer in enumerate(lora_layer_list):
                if lora_name in lora_layer.loras_:
                    lora_weight_dict[layer_prefix_name +
                                     f"{lora_layer_name_list[idx]}.lora_A.weight"] = lora_layer.loras_[lora_name].lora_a_
                    lora_weight_dict[layer_prefix_name +
                                     f"{lora_layer_name_list[idx]}.lora_B.weight"] = lora_layer.loras_[lora_name].lora_b_
                elif lora_name in transformer_layer.ffn_.moes_:
                    moe_layer_prefix_name = f"mixlora.layers.{transformer_layer.layer_id_}."
                    for expert_idx in range(transformer_layer.ffn_.moes_[lora_name].experts_):
                        moe_lora_name = f"moe.{lora_name}.experts.{expert_idx}"
                        if moe_lora_name in lora_layer.loras_:
                            lora_weight_dict[
                                moe_layer_prefix_name
                                + f"experts.{expert_idx}."
                                + f"{lora_layer_name_list[idx]}.lora_A.weight"
                            ] = lora_layer.loras_[moe_lora_name].lora_a_
                            lora_weight_dict[
                                moe_layer_prefix_name
                                + f"experts.{expert_idx}."
                                + f"{lora_layer_name_list[idx]}.lora_B.weight"
                            ] = lora_layer.loras_[moe_lora_name].lora_b_

                    lora_weight_dict[
                        moe_layer_prefix_name + "gate.weight"
                    ] = transformer_layer.ffn_.moes_[lora_name].gate_.weight

        return lora_weight_dict

    def sequential_module(self) -> torch.nn.Sequential:
        seq_module = OrderedDict()

        seq_module.update(
            {"embedding": LlamaSequentialWrapper(self.token_embedding_)})
        seq_module.move_to_end("embedding")

        for index, layer in enumerate(self.layers_):
            layer_name = f"layer{index}"
            seq_module.update({layer_name: LlamaSequentialWrapper(layer)})
            seq_module.move_to_end(layer_name)

        seq_module.update({"norm": LlamaSequentialWrapper(self.norm_)})
        seq_module.move_to_end("norm")

        seq_module.update({"output": LlamaSequentialWrapper(self.output_)})
        seq_module.move_to_end("output")

        return torch.nn.Sequential(seq_module)

    def load_adapter_weight(self, path: str, adapter_name: str = None):
        if adapter_name is None:
            adapter_name = path
        if not os.path.exists(path):
            path = snapshot_download(repo_id=path, repo_type="model")
        with open(path + os.sep + "adapter_config.json", 'r', encoding='utf8') as fp:
            lora_config = lora_config_factory(json.load(fp))
        lora_config.adapter_name_ = adapter_name
        lora_weight = torch.load(
            path + os.sep + "adapter_model.bin", map_location=self.device_)
        self.init_lora_layer_weight(lora_config, lora_weight)
        return adapter_name

    def save_adapter_weight(self, path: str, dir_suffix=""):
        for lora_name, lora_config in self.adapter_configs_.items():
            lora_output_dir = path + os.sep + lora_name
            if dir_suffix != "":
                lora_output_dir += os.sep + \
                    lora_name + "_" + dir_suffix

            if not os.path.exists(lora_output_dir):
                os.makedirs(lora_output_dir)

            lora_weight_dict = self.get_lora_weight_dict(
                lora_name)

            lora_config_dict = lora_config.export()

            torch.save(lora_weight_dict, lora_output_dir +
                       os.sep + "adapter_model.bin")

            with open(lora_output_dir + os.sep + "adapter_config.json", "w") as f:
                json.dump(lora_config_dict, f, indent=4)
