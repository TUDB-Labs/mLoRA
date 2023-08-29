import sys
import torch

from aspen import LlamaModel, Linear, RMSNorm
from transformers import LlamaForCausalLM


def load_llama_7b_weight(model: LlamaModel, llama_model_path: str, device: str):
    weight = torch.load(llama_model_path, map_location=torch.device(device))

    for layer_name in weight:
        w: torch.Tensor = weight[layer_name]
        w.requires_grad_(False)

        if "layers" in layer_name:
            layer_name = layer_name[len("layers."):]
            layer_id = int(layer_name[:layer_name.find(".")])
            if "wq" in layer_name:
                model.layers_[layer_id].wq_ = Linear(w)
            elif "wk" in layer_name:
                model.layers_[layer_id].wk_ = Linear(w)
            elif "wv" in layer_name:
                model.layers_[layer_id].wv_ = Linear(w)
            elif "wo" in layer_name:
                model.layers_[layer_id].wo_ = Linear(w)
            elif "w1" in layer_name:
                model.layers_[layer_id].w1_ = Linear(w)
            elif "w2" in layer_name:
                model.layers_[layer_id].w2_ = Linear(w)
            elif "w3" in layer_name:
                model.layers_[layer_id].w3_ = Linear(w)
            elif "attention_norm" in layer_name:
                model.layers_[layer_id].attention_norm_ = RMSNorm(
                    w, model.norm_eps_)
            elif "ffn_norm" in layer_name:
                model.layers_[layer_id].ffn_norm_ = RMSNorm(
                    w, model.norm_eps_)
            else:
                print(f"Not use layer {layer_name}.", file=sys.stderr)
        elif "tok_embeddings" in layer_name:
            model.token_embedding_ = w
        elif "norm.weight" in layer_name:
            model.norm_ = RMSNorm(w, model.norm_eps_)
        elif "output.weight" in layer_name:
            model.output_ = w.to(torch.float32)
        else:
            print(f"Not use layer {layer_name}.", file=sys.stderr)


def load_llama_tf_weight(model: LlamaModel, llama_model_path: str, dev: str):
    weight = LlamaForCausalLM.from_pretrained(
        llama_model_path, device_map=dev).state_dict()

    for layer_name in weight:
        w: torch.Tensor = weight[layer_name]
        w.requires_grad_(False)

        if "model.layers" in layer_name:
            layer_name = layer_name[len("model.layers."):]
            layer_id = int(layer_name[:layer_name.find(".")])
            if "self_attn.q_proj" in layer_name:
                model.layers_[layer_id].wq_ = Linear(w)
            elif "self_attn.k_proj" in layer_name:
                model.layers_[layer_id].wk_ = Linear(w)
            elif "self_attn.v_proj" in layer_name:
                model.layers_[layer_id].wv_ = Linear(w)
            elif "self_attn.o_proj" in layer_name:
                model.layers_[layer_id].wo_ = Linear(w)
            elif "mlp.gate_proj" in layer_name:
                model.layers_[layer_id].w1_ = Linear(w)
            elif "mlp.down_proj" in layer_name:
                model.layers_[layer_id].w2_ = Linear(w)
            elif "mlp.up_proj" in layer_name:
                model.layers_[layer_id].w3_ = Linear(w)
            elif "input_layernorm" in layer_name:
                model.layers_[layer_id].attention_norm_ = RMSNorm(
                    w, model.norm_eps_)
            elif "post_attention_layernorm" in layer_name:
                model.layers_[layer_id].ffn_norm_ = RMSNorm(
                    w, model.norm_eps_)
            else:
                print(
                    f"Not use layer model.layers.{layer_name}.", file=sys.stderr)
        elif "embed_tokens" in layer_name:
            model.token_embedding_ = w
        elif "norm.weight" in layer_name:
            model.norm_ = RMSNorm(w, model.norm_eps_)
        elif "lm_head.weight" in layer_name:
            model.output_ = w.to(torch.float32)
        else:
            print(f"Not use layer {layer_name}.", file=sys.stderr)


def load_random_lora_7b_weight(model: LlamaModel, adapter_name: str, r: int, dim: int, target_module: str, device: str) -> None:
    norm_mean = 0
    norm_std = 1e-3
    target_module_name_list = ["q_proj", "k_proj", "v_proj", "o_proj", "w1_proj", "w2_proj", "w3_proj"]
    for transformer_layer in model.layers_:
        target_layer_list = [transformer_layer.wq_, transformer_layer.wk_,
                             transformer_layer.wv_, transformer_layer.wo_,
                             transformer_layer.w1_, transformer_layer.w2_,
                             transformer_layer.w3_]
        for idx, module_name in enumerate(target_module_name_list):
            if module_name in target_module and target_module[module_name]:
                lora_a_weight = torch.normal(
                    mean=norm_mean, std=norm_std, size=(r, dim), device=device, requires_grad=True, dtype=torch.float32)
                lora_b_weight = torch.normal(
                    mean=norm_mean, std=norm_std, size=(dim, r), device=device, requires_grad=True, dtype=torch.float32)
                target_layer_list[idx].set_lora_layer_weight(
                    adapter_name, "lora_A", lora_a_weight)
                target_layer_list[idx].set_lora_layer_weight(
                    adapter_name, "lora_B", lora_b_weight)


def save_lora_model(model: LlamaModel, path: str, lora_name: str):
    lora_weight_dict = {}
    for idx, transformer_layer in enumerate(model.layers_):
        layer_prefix_name = "base_model.model.model.layers." + \
            str(idx) + "." + "self_attn."
        lora_layer_list = [transformer_layer.wq_, transformer_layer.wk_,
                           transformer_layer.wv_, transformer_layer.wo_,
                           transformer_layer.w1_, transformer_layer.w2_,
                           transformer_layer.w3_]
        lora_layer_name_list = ["q_proj", "k_proj", "v_proj", "o_proj", "w1_proj", "w2_proj", "w3_proj"]
        for idx, lora_layer in enumerate(lora_layer_list):
            if lora_name in lora_layer.loras_:
                lora_weight_dict[layer_prefix_name +
                                 f"{lora_layer_name_list[idx]}.lora_A.weight"] = lora_layer.loras_[lora_name].lora_a_
                lora_weight_dict[layer_prefix_name +
                                 f"{lora_layer_name_list[idx]}.lora_B.weight"] = lora_layer.loras_[lora_name].lora_b_

    torch.save(lora_weight_dict, path)
