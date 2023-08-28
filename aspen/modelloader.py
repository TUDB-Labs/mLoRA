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
            model.output_ = w
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
            model.output_ = w
        else:
            print(f"Not use layer {layer_name}.", file=sys.stderr)


def load_alpaca_lora_7b_weight(model: LlamaModel, lora_model_path: str, adapter_name: str, device: str):
    lora_weight = torch.load(
        lora_model_path, map_location=torch.device(device))
    for layer_name in lora_weight:
        w: torch.Tensor = lora_weight[layer_name].to(torch.float16)
        w.requires_grad_(True)

        layer_name = layer_name[len("base_model.model.model.layers."):]
        layer_id = int(layer_name[:layer_name.find(".")])
        lora_name = ""
        if "lora_A" in layer_name:
            lora_name = "lora_A"
        elif "lora_B" in layer_name:
            lora_name = "lora_B"

        if "q_proj" in layer_name:
            model.layers_[layer_id].wq_.update_lora_weight(
                adapter_name, lora_name, w)
            model.layers_[layer_id].wq_.use_adapter_ = True
        elif "k_proj" in layer_name:
            model.layers_[layer_id].wk_.update_lora_weight(
                adapter_name, lora_name, w)
            model.layers_[layer_id].wk_.use_adapter_ = True
        elif "v_proj" in layer_name:
            model.layers_[layer_id].wv_.update_lora_weight(
                adapter_name, lora_name, w)
            model.layers_[layer_id].wv_.use_adapter_ = True
        elif "o_proj" in layer_name:
            model.layers_[layer_id].wo_.update_lora_weight(
                adapter_name, lora_name, w)
            model.layers_[layer_id].wo_.use_adapter_ = True
        else:
            print(f"Not user layer {layer_name}")


def load_random_lora_7b_weight(model: LlamaModel, adapter_name: str, r: int, dim: int, target_module: str, device: str) -> None:
    norm_mean = 0
    norm_std = 1e-3
    for layer in model.layers_:
        if target_module["q_proj"] is True:
            wq_lora_a_weight = torch.normal(
                mean=norm_mean, std=norm_std, size=(r, dim), device=device, requires_grad=True, dtype=torch.float16)
            wq_lora_b_weight = torch.normal(
                mean=norm_mean, std=norm_std, size=(dim, r), device=device, requires_grad=True, dtype=torch.float16)
            layer.wq_.update_lora_weight(
                adapter_name, "lora_A", wq_lora_a_weight)
            layer.wq_.update_lora_weight(
                adapter_name, "lora_B", wq_lora_b_weight)
            layer.wq_.use_adapter_ = True

        if target_module["k_proj"] is True:
            wk_lora_a_weight = torch.normal(
                mean=norm_mean, std=norm_std, size=(r, dim), device=device, requires_grad=True, dtype=torch.float16)
            wk_lora_b_weight = torch.normal(
                mean=norm_mean, std=norm_std, size=(dim, r), device=device, requires_grad=True, dtype=torch.float16)
            layer.wk_.update_lora_weight(
                adapter_name, "lora_A", wk_lora_a_weight)
            layer.wk_.update_lora_weight(
                adapter_name, "lora_B", wk_lora_b_weight)
            layer.wk_.use_adapter_ = True

        if target_module["v_proj"] is True:
            wv_lora_a_weight = torch.normal(
                mean=norm_mean, std=norm_std, size=(r, dim), device=device, requires_grad=True, dtype=torch.float16)
            wv_lora_b_weight = torch.normal(
                mean=norm_mean, std=norm_std, size=(dim, r), device=device, requires_grad=True, dtype=torch.float16)
            layer.wv_.update_lora_weight(
                adapter_name, "lora_A", wv_lora_a_weight)
            layer.wv_.update_lora_weight(
                adapter_name, "lora_B", wv_lora_b_weight)
            layer.wv_.use_adapter_ = True

        if target_module["o_proj"] is True:
            wo_lora_a_weight = torch.normal(
                mean=norm_mean, std=norm_std, size=(r, dim), device=device, requires_grad=True, dtype=torch.float16)
            wo_lora_b_weight = torch.normal(
                mean=norm_mean, std=norm_std, size=(dim, r), device=device, requires_grad=True, dtype=torch.float16)
            layer.wo_.update_lora_weight(
                adapter_name, "lora_A", wo_lora_a_weight)
            layer.wo_.update_lora_weight(
                adapter_name, "lora_B", wo_lora_b_weight)
            layer.wo_.use_adapter_ = True


def save_lora_model(model: LlamaModel, path: str, lora_name: str):
    lora_weight_dict = {}
    for idx, layer in enumerate(model.layers_):
        layer_prefix_name = "base_model.model.model.layers." + \
            str(idx) + "." + "self_attn."
        if lora_name in layer.wq_.lora_a_:
            lora_weight_dict[layer_prefix_name +
                             "q_proj.lora_A.weight"] = layer.wq_.lora_a_[lora_name]
        if lora_name in layer.wq_.lora_b_:
            lora_weight_dict[layer_prefix_name +
                             "q_proj.lora_B.weight"] = layer.wq_.lora_b_[lora_name]
        if lora_name in layer.wk_.lora_a_:
            lora_weight_dict[layer_prefix_name +
                             "k_proj.lora_A.weigth"] = layer.wk_.lora_a_[lora_name]
        if lora_name in layer.wk_.lora_b_:
            lora_weight_dict[layer_prefix_name +
                             "k_proj.lora_B.weight"] = layer.wk_.lora_b_[lora_name]
        if lora_name in layer.wv_.lora_a_:
            lora_weight_dict[layer_prefix_name +
                             "v_proj.lora_A.weight"] = layer.wv_.lora_a_[lora_name]
        if lora_name in layer.wv_.lora_b_:
            lora_weight_dict[layer_prefix_name +
                             "v_proj.lora_B.weight"] = layer.wv_.lora_b_[lora_name]
        if lora_name in layer.wo_.lora_a_:
            lora_weight_dict[layer_prefix_name +
                             "o_proj.lora_A.weight"] = layer.wo_.lora_a_[lora_name]
        if lora_name in layer.wo_.lora_b_:
            lora_weight_dict[layer_prefix_name +
                             "o_proj.lora_B.weight"] = layer.wo_.lora_b_[lora_name]

    torch.save(lora_weight_dict, path)
