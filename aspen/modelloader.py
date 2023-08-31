from aspen.model import LlamaModel, Linear, RMSNorm

import os
import json
import torch

from typing import Dict
from transformers import LlamaForCausalLM


def load_llama_tf_weight(model: LlamaModel, llama_model_path: str, dev: str, load_in_8bit: bool = False):
    llama_model = LlamaForCausalLM.from_pretrained(
        llama_model_path, load_in_8bit=load_in_8bit, device_map=dev)

    model.token_embedding_ = llama_model.model.embed_tokens.weight.to(device=dev)
    model.output_ = llama_model.lm_head.weight.to(dtype=torch.float32, device=dev)
    model.norm_ = RMSNorm(llama_model.model.norm.weight.to(device=dev), model.norm_eps_)

    for idx, layer in enumerate(llama_model.model.layers):
        model.layers_[idx].wq_ = Linear(layer.self_attn.q_proj, device=dev)
        model.layers_[idx].wk_ = Linear(layer.self_attn.k_proj, device=dev)
        model.layers_[idx].wv_ = Linear(layer.self_attn.v_proj, device=dev)
        model.layers_[idx].wo_ = Linear(layer.self_attn.o_proj, device=dev)
        model.layers_[idx].w1_ = Linear(layer.mlp.gate_proj, device=dev)
        model.layers_[idx].w2_ = Linear(layer.mlp.down_proj, device=dev)
        model.layers_[idx].w3_ = Linear(layer.mlp.up_proj, device=dev)
        model.layers_[idx].attention_norm_ = RMSNorm(layer.input_layernorm.weight.to(device=dev), model.norm_eps_)
        model.layers_[idx].ffn_norm_ = RMSNorm(layer.post_attention_layernorm.weight.to(device=dev), model.norm_eps_)


def save_lora_model(model: LlamaModel, config: Dict[str, str], dir_suffix=""):
    for lora_config in config["lora"]:
        lora_name = lora_config["name"]
        lora_output_dir = lora_config["output"]
        if dir_suffix != "":
            lora_output_dir += os.sep + \
                lora_config["output"] + "_" + dir_suffix

        if not os.path.exists(lora_output_dir):
            os.makedirs(lora_output_dir)

        lora_weight_dict = {}
        target_modules = []
        for idx, transformer_layer in enumerate(model.layers_):
            layer_prefix_name = "base_model.model.model.layers." + \
                str(idx) + "." + "self_attn."
            lora_layer_list = [transformer_layer.wq_, transformer_layer.wk_,
                               transformer_layer.wv_, transformer_layer.wo_,
                               transformer_layer.w1_, transformer_layer.w2_,
                               transformer_layer.w3_]
            lora_layer_name_list = [
                "q_proj", "k_proj", "v_proj", "o_proj", "w1_proj", "w2_proj", "w3_proj"]
            for idx, lora_layer in enumerate(lora_layer_list):
                if lora_name in lora_layer.loras_:
                    if lora_layer_name_list[idx] not in target_modules:
                        target_modules.append(lora_layer_name_list[idx])
                    lora_weight_dict[layer_prefix_name +
                                     f"{lora_layer_name_list[idx]}.lora_A.weight"] = lora_layer.loras_[lora_name].lora_a_
                    lora_weight_dict[layer_prefix_name +
                                     f"{lora_layer_name_list[idx]}.lora_B.weight"] = lora_layer.loras_[lora_name].lora_b_

        torch.save(lora_weight_dict, lora_output_dir +
                   os.sep + "adapter_model.bin")

        adapter_config = {}
        adapter_config["lora_alpha"] = lora_config["alpha"]
        adapter_config["lora_dropout"] = lora_config["dropout"]
        adapter_config["r"] = lora_config["r"]
        adapter_config["peft_type"] = "LORA"
        adapter_config["task_type"] = "CAUSAL_LM"
        adapter_config["bias"] = "none"
        adapter_config["target_modules"] = target_modules

        with open(lora_output_dir + os.sep + "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=4)
