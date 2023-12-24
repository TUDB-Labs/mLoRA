from mlora.modelargs import LLMModelArgs, MultiLoraBatchData
from mlora.checkpoint import CheckpointRecomputeFunction
from mlora.model import (
    repeat_kv,
    apply_rotary_emb,
    precompute_rope_angle,
    precompute_mask,
)
from mlora.model import LLMModel, RMSNorm
from mlora.LoraLiner import Linear

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import xformers.ops
import xformers.ops.fmha.attn_bias
from transformers import LlamaForCausalLM
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict
import os
import json


class Embedding(torch.nn.Module):
    def __init__(self, embedding: torch.Tensor, pad_token: int):
        super().__init__()
        self.token_embedding_: torch.Tensor = embedding
        self.padding_idx_: int = pad_token

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        data = F.embedding(tokens, self.token_embedding_, padding_idx=self.padding_idx_)
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
        return data * torch.rsqrt(+self.norm_eps_)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        input_dtype = data.dtype
        v = data.to(torch.float32).pow(2).mean(-1, keepdim=True)
        data = data * torch.rsqrt(v + self.norm_eps_)

        return (self.weight_ * data).to(input_dtype)


class MixMoELayer:
    gate_: torch.nn.Linear = None
    experts_: int = 8
    topk_: int = 2


class MixTransformer(torch.nn.Module):
    def __init__(self, layer_id: int, args: LLMModelArgs):
        super().__init__()
        # attention
        self.wq_: Linear = None  # dim * dim
        self.wk_: Linear = None  # dim * dim
        self.wv_: Linear = None  # dim * dim
        self.wo_: Linear = None  # dim * dim
        # feed forward
        self.w1_: Linear = None  # also gate FNN * dim
        self.w2_: Linear = None  # also down dim * FNN
        self.w3_: Linear = None  # also up   FNN * dim
        # norm
        self.attention_norm_: RMSNorm = None  # dim
        self.ffn_norm_: RMSNorm = None  # dim
        # other arg
        self.layer_id_ = layer_id
        self.norm_eps_ = args.norm_eps_
        self.n_heads_ = args.n_heads_
        self.n_kv_heads_ = args.n_kv_heads_
        self.n_rep_ = self.n_heads_ // self.n_kv_heads_
        self.head_dim_ = args.dim_ // args.n_heads_
        # moe
        self.moes_: Dict[str, MixMoELayer] = {}

    def init_moe_layer_weight(
        self,
        moe_name: str,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        moe_experts: int,
        moe_topk: int,
        target: Dict[str, bool],
        weight: Optional[torch.Tensor],
    ):
        # Inject LoRA configs into Attention Layer
        attention_layer_list = [self.wk_, self.wq_, self.wv_, self.wo_]
        attention_layer_name_list = ["k_proj", "q_proj", "v_proj", "o_proj"]

        for idx, layer_name in enumerate(attention_layer_name_list):
            if layer_name in target and target[layer_name]:
                lora_a = None
                lora_b = None
                if weight is not None:
                    lora_a_name = f"mixlora.layers.{self.layer_id_}.self_attn.{layer_name}.lora_A.weight"
                    lora_b_name = f"mixlora.layers.{self.layer_id_}.self_attn.{layer_name}.lora_B.weight"
                    if lora_a_name not in weight:
                        raise f"can not found the layer {lora_a_name} in model"
                    if lora_b_name not in weight:
                        raise f"can not found the layer {lora_b_name} in model"
                    lora_a = weight[lora_a_name]
                    lora_b = weight[lora_b_name]

                attention_layer_list[idx].init_lora_weight(
                    moe_name, lora_r, lora_alpha, lora_dropout, lora_a, lora_b
                )

        # Inject LoRA configs into FFN layer
        moe_layer = MixMoELayer()
        moe_layer.gate_ = torch.nn.Linear(
            self.n_heads_ * self.head_dim_,
            moe_experts,
            bias=False,
            device=self.w1_.device_,
        )
        moe_layer.experts_ = moe_experts
        moe_layer.topk_ = moe_topk

        if weight is not None:
            with torch.no_grad():
                moe_layer.gate_.weight.copy_(
                    weight[f"mixlora.layers.{self.layer_id_}.gate.weight"]
                )

        self.moes_[moe_name] = moe_layer

        ffn_layer_list = [self.w1_, self.w2_, self.w3_]
        ffn_layer_name_list = ["w1_proj", "w2_proj", "w3_proj"]
        for idx, layer_name in enumerate(ffn_layer_name_list):
            if layer_name in target and target[layer_name]:
                for expert_idx in range(moe_layer.experts_):
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

                    ffn_layer_list[idx].init_lora_weight(
                        f"moe.{moe_name}.experts.{expert_idx}",
                        lora_r,
                        lora_alpha,
                        lora_dropout,
                        lora_a,
                        lora_b,
                    )

    # @torch.compile
    def forward(
        self,
        data: torch.Tensor,
        mask: torch.Tensor,
        rope_angle: Tuple[torch.Tensor, torch.Tensor],
        input_args: MultiLoraBatchData,
    ):
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

        # for llama2 need to repeat the heads
        # before dim: batch_size, seq_len, n_kv_head, head_dim
        # after dim: batch_size, seq_len, n_head, head_dim
        xk = repeat_kv(xk, self.n_rep_)
        xv = repeat_kv(xv, self.n_rep_)

        attention_score = xformers.ops.memory_efficient_attention(xq, xk, xv, mask)
        attention_score = attention_score.view(batch_size, max_seq_len, -1)

        # get output attention score
        data = data + self.wo_.forward(attention_score, input_args)

        # feed forward fully connected
        score_norm_data = self.ffn_norm_.forward(data)
        common_w1 = self.w1_.weight_.forward(score_norm_data)
        common_w3 = self.w3_.weight_.forward(score_norm_data)

        final_ffn_output = None
        for lora_config in input_args.lora_batch_data_config_:
            moe_name = lora_config.adapter_name_
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            if moe_name == "" or moe_name not in self.moes_:
                continue

            moe_layer = self.moes_[moe_name]

            norm_data = score_norm_data[start_idx:end_idx]
            w1_data = common_w1[start_idx:end_idx]
            w3_data = common_w3[start_idx:end_idx]

            # routing to experts
            router_logits = moe_layer.gate_.forward(norm_data)
            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            routing_weights, selected_experts = torch.topk(
                routing_weights, moe_layer.topk_, dim=-1
            )
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

            # mix of experts
            final_hidden_states = None
            for expert_idx in range(moe_layer.experts_):
                lora_name = f"moe.{moe_name}.experts.{expert_idx}"
                if lora_name in self.w1_.loras_:
                    w1 = w1_data + self.w1_.loras_[lora_name].forward(norm_data)
                else:
                    w1 = w1_data

                if lora_name in self.w3_.loras_:
                    w3 = w3_data + self.w3_.loras_[lora_name].forward(norm_data)
                else:
                    w3 = w3_data

                silu_result = F.silu(w1) * w3
                if lora_name in self.w2_.loras_:
                    hidden_state = self.w2_.weight_.forward(
                        silu_result
                    ) + self.w2_.loras_[lora_name].forward(silu_result)
                else:
                    hidden_state = self.w2_.weight_.forward(silu_result)

                expert_mask = selected_experts == expert_idx
                expert_weights = (routing_weights * expert_mask).sum(
                    dim=-1, keepdim=True
                )
                current_hidden_states = hidden_state.mul_(expert_weights)
                if final_hidden_states is None:
                    final_hidden_states = current_hidden_states
                else:
                    final_hidden_states.add_(current_hidden_states)

            if final_ffn_output is None:
                final_ffn_output = final_hidden_states
            else:
                final_ffn_output = torch.cat(
                    [final_ffn_output, final_hidden_states], dim=0
                )

        return data + final_ffn_output


class MixSequentialWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.wrapper_module_ = module

    def name(self) -> str:
        return type(self.wrapper_module_).__name__

    def forward(self, input: Tuple) -> Tuple:
        module_name = self.name()

        if (
            module_name == "Embedding"
            or module_name == "RMSNormLayer"
            or module_name == "OutputLayer"
        ):
            output = self.wrapper_module_.forward(input[0])
            return (output,) + input[1:]
        elif module_name == "MixTransformer":
            if input[-1]:
                output = CheckpointRecomputeFunction.apply(
                    self.wrapper_module_.forward, *input[:-1]
                )
            else:
                output = self.wrapper_module_.forward(*input[:-1])
            return (output,) + input[1:]
        else:
            raise f"module invalid: {module_name}"


class MixModel:
    def __init__(self, args: LLMModelArgs):
        # weight
        self.token_embedding_: Embedding = None

        self.layers_: List[MixTransformer] = []
        for layer_id in range(args.n_layers_):
            self.layers_.append(MixTransformer(layer_id, args))

        self.norm_: RMSNormLayer = None  # dim
        self.output_: OutputLayer = None  # vocab size * dim

        # cos and sin
        self.rope_angle_: Tuple[torch.Tensor, torch.Tensor] = precompute_rope_angle(
            args.dim_ // args.n_heads_, args.max_seq_len_, args.device
        )

        self.norm_eps_ = args.norm_eps_

        self.device_ = args.device
        self.n_heads_ = args.n_heads_
        self.vocab_size_ = args.vocab_size_
        self.pad_token_id_ = args.pad_token_id_
        self.dim_ = args.dim_

        # need to set
        self.eos_token_id_ = -1

    # train model or inference model: output is probs
    def forward(self, input: MultiLoraBatchData) -> torch.Tensor:
        tokens = torch.tensor(input.batch_tokens_, dtype=torch.int64).to(self.device_)

        # only for train
        mask = precompute_mask(input, self.n_heads_, self.device_)

        seq_module = self.sequential_module()

        if input.inference_model_:
            data = (tokens, mask, self.rope_angle_, input, False)
        else:
            data = (tokens, mask, self.rope_angle_, input, True)

        for seq_layer in seq_module:
            data = seq_layer.forward(data)

        return data[0]

    def init_moe_weight(
        self,
        moe_name: str,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        moe_experts: int,
        moe_topk: int,
        target: Dict[str, bool],
        weight: Optional[Dict[str, torch.Tensor]],
    ):
        for transformer_layer in self.layers_:
            transformer_layer.init_moe_layer_weight(
                moe_name=moe_name,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                moe_experts=moe_experts,
                moe_topk=moe_topk,
                target=target,
                weight=weight,
            )

    def from_pretrained(
        path: str,
        device: str,
        bits: int = None,
        fp16: bool = True,
        bf16: bool = True,
        double_quant: bool = True,
        quant_type: str = "nf4",
        log_fn=None,
    ) -> LLMModel:
        if bits in [4, 8]:
            if log_fn is not None:
                log_fn("Loading model with quantization, bits = %i" % bits)
            from transformers import BitsAndBytesConfig

            compute_dtype = (
                torch.float16 if fp16 else (torch.bfloat16 if bf16 else torch.float32)
            )
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
                torch_dtype=(
                    torch.float32
                    if fp16
                    else (torch.bfloat16 if bf16 else torch.float32)
                ),
            )
        else:
            llama_model = LlamaForCausalLM.from_pretrained(
                path, device_map=device, torch_dtype=torch.float32
            )

        llama_args = LLMModelArgs()
        llama_args.dim_ = llama_model.config.hidden_size
        llama_args.n_heads_ = llama_model.config.num_attention_heads
        llama_args.n_kv_heads_ = (
            llama_args.n_heads_
            if not hasattr(llama_model.config, "num_key_value_heads")
            else llama_model.config.num_key_value_heads
        )
        llama_args.n_layers_ = llama_model.config.num_hidden_layers
        llama_args.norm_eps_ = llama_model.config.rms_norm_eps
        llama_args.vocab_size_ = llama_model.config.vocab_size
        llama_args.max_seq_len_ = (
            4096
            if not hasattr(llama_model.config, "max_sequence_length")
            else llama_model.config.max_sequence_length
        )
        llama_args.pad_token_id_ = -1
        llama_args.device = device

        model = MixModel(llama_args)

        embedding_weight = llama_model.model.embed_tokens.weight.to(
            device=device
        ).requires_grad_(False)
        model.token_embedding_ = Embedding(embedding_weight, llama_args.pad_token_id_)

        output_weight = llama_model.lm_head.weight.to(
            dtype=torch.float32, device=device
        ).requires_grad_(False)
        model.output_ = OutputLayer(output_weight)

        norm_weight = llama_model.model.norm.weight.to(device=device).requires_grad_(
            False
        )
        model.norm_ = RMSNormLayer(norm_weight, model.norm_eps_)

        for idx, layer in enumerate(llama_model.model.layers):
            model.layers_[idx].wq_ = Linear(layer.self_attn.q_proj, device=device)
            model.layers_[idx].wk_ = Linear(layer.self_attn.k_proj, device=device)
            model.layers_[idx].wv_ = Linear(layer.self_attn.v_proj, device=device)
            model.layers_[idx].wo_ = Linear(layer.self_attn.o_proj, device=device)
            model.layers_[idx].w1_ = Linear(layer.mlp.gate_proj, device=device)
            model.layers_[idx].w2_ = Linear(layer.mlp.down_proj, device=device)
            model.layers_[idx].w3_ = Linear(layer.mlp.up_proj, device=device)
            model.layers_[idx].attention_norm_ = RMSNorm(
                layer.input_layernorm.weight.to(device=device).requires_grad_(False),
                model.norm_eps_,
            )
            model.layers_[idx].ffn_norm_ = RMSNorm(
                layer.post_attention_layernorm.weight.to(device=device).requires_grad_(
                    False
                ),
                model.norm_eps_,
            )

        return model

    def get_train_paramas(
        self, config: Dict[str, str]
    ) -> Dict[str, List[torch.Tensor]]:
        train_paramas = {}

        for transformer_layer in self.layers_:
            for moe_config in config["lora"]:
                moe_name = moe_config["name"]
                if moe_name not in train_paramas:
                    train_paramas[moe_name] = []

                lora_layer_list = [
                    transformer_layer.wq_.loras_,
                    transformer_layer.wk_.loras_,
                    transformer_layer.wv_.loras_,
                    transformer_layer.wo_.loras_,
                    transformer_layer.w1_.loras_,
                    transformer_layer.w2_.loras_,
                    transformer_layer.w3_.loras_,
                ]

                for lora_layer in lora_layer_list:
                    if moe_name in lora_layer:
                        train_paramas[moe_name].append(lora_layer[moe_name].lora_a_)
                        train_paramas[moe_name].append(lora_layer[moe_name].lora_b_)
                    else:
                        for expert_idx in range(moe_config["experts"]):
                            lora_name = f"moe.{moe_name}.experts.{expert_idx}"
                            if lora_name in lora_layer:
                                train_paramas[moe_name].append(
                                    lora_layer[lora_name].lora_a_
                                )
                                train_paramas[moe_name].append(
                                    lora_layer[lora_name].lora_b_
                                )

        return train_paramas

    def get_moe_weight_dict(
        self, moe_name: str
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        moe_weight_dict = {}
        target_modules = []
        for idx, transformer_layer in enumerate(self.layers_):
            layer_prefix_name = f"mixlora.layers.{idx}."
            lora_layer_list = [
                transformer_layer.wq_,
                transformer_layer.wk_,
                transformer_layer.wv_,
                transformer_layer.wo_,
                transformer_layer.w1_,
                transformer_layer.w2_,
                transformer_layer.w3_,
            ]
            lora_layer_name_list = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "w1_proj",
                "w2_proj",
                "w3_proj",
            ]
            for idx, lora_layer in enumerate(lora_layer_list):
                if moe_name in lora_layer.loras_:
                    if lora_layer_name_list[idx] not in target_modules:
                        target_modules.append(lora_layer_name_list[idx])
                    moe_weight_dict[
                        layer_prefix_name
                        + f"self_attn.{lora_layer_name_list[idx]}.lora_A.weight"
                    ] = lora_layer.loras_[moe_name].lora_a_
                    moe_weight_dict[
                        layer_prefix_name
                        + f"self_attn.{lora_layer_name_list[idx]}.lora_B.weight"
                    ] = lora_layer.loras_[moe_name].lora_b_
                else:
                    moe_layer = transformer_layer.moes_[moe_name]
                    for expert_idx in range(moe_layer.experts_):
                        lora_name = f"moe.{moe_name}.experts.{expert_idx}"
                        if lora_name in lora_layer.loras_:
                            if lora_layer_name_list[idx] not in target_modules:
                                target_modules.append(lora_layer_name_list[idx])
                            moe_weight_dict[
                                layer_prefix_name
                                + f"experts.{expert_idx}."
                                + f"{lora_layer_name_list[idx]}.lora_A.weight"
                            ] = lora_layer.loras_[lora_name].lora_a_
                            moe_weight_dict[
                                layer_prefix_name
                                + f"experts.{expert_idx}."
                                + f"{lora_layer_name_list[idx]}.lora_B.weight"
                            ] = lora_layer.loras_[lora_name].lora_b_

                    moe_weight_dict[
                        layer_prefix_name + "gate.weight"
                    ] = moe_layer.gate_.weight

        return moe_weight_dict, target_modules

    def sequential_module(self) -> torch.nn.Sequential:
        seq_module = OrderedDict()

        seq_module.update({"embedding": MixSequentialWrapper(self.token_embedding_)})
        seq_module.move_to_end("embedding")

        for index, layer in enumerate(self.layers_):
            layer_name = f"layer{index}"
            seq_module.update({layer_name: MixSequentialWrapper(layer)})
            seq_module.move_to_end(layer_name)

        seq_module.update({"norm": MixSequentialWrapper(self.norm_)})
        seq_module.move_to_end("norm")

        seq_module.update({"output": MixSequentialWrapper(self.output_)})
        seq_module.move_to_end("output")

        return torch.nn.Sequential(seq_module)


def save_mixlora_model(model: MixModel, config: Dict[str, str], dir_suffix=""):
    for moe_config in config["lora"]:
        moe_name = moe_config["name"]
        output_dir = moe_config["output"]
        if dir_suffix != "":
            output_dir += os.sep + moe_config["output"] + "_" + dir_suffix

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        weight_dict, target_modules = model.get_moe_weight_dict(moe_name)
        torch.save(weight_dict, output_dir + os.sep + "mixlora_model.bin")

        mixlora_config = {}
        mixlora_config["lora_alpha"] = moe_config["alpha"]
        mixlora_config["lora_dropout"] = moe_config["dropout"]
        mixlora_config["r"] = moe_config["r"]
        mixlora_config["peft_type"] = "LORA"
        mixlora_config["task_type"] = "CAUSAL_LM"
        mixlora_config["bias"] = "none"
        mixlora_config["target_modules"] = target_modules
        mixlora_config["experts"] = moe_config["experts"]
        mixlora_config["topk"] = moe_config["topk"]

        with open(output_dir + os.sep + "mixlora_config.json", "w") as f:
            json.dump(mixlora_config, f, indent=4)