from aspen.modelargs import LlamaModelArgs, MultiLoraBatchData

import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import einops
import bitsandbytes
import xformers.ops
import xformers.ops.fmha.attn_bias
from transformers import LlamaForCausalLM
from typing import List, Dict, Tuple


def precompute_rope_angle(dim: int, seq_len: int, device: str, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    angles = 1.0 / (theta ** (torch.arange(0, dim, 2).to(device)
                              [: (dim // 2)].to(torch.float) / dim))
    seq = torch.arange(seq_len, device=angles.device)
    emb = torch.outer(seq, angles).float()
    emb = einops.repeat(emb, "... n -> ... (n r)", r=2)
    # cos(angle), sin(angle)
    return (emb.cos().to(torch.float32), emb.sin().to(torch.float32))


def precompute_mask(input: MultiLoraBatchData, n_head: int, device: str) -> torch.Tensor:
    mask = torch.full((len(input.prompts_), n_head,
                      input.batch_seq_len_, input.batch_seq_len_), float("-inf"))
    mask = torch.triu(mask, diagonal=1).to(torch.float32).cuda(device)

    for idx, _ in enumerate(input.prompts_):
        zero_len = input.tokens_len_without_pad_[idx]
        inf_len = input.batch_seq_len_ - zero_len
        if input.expand_right_:
            mask[idx] += torch.tensor([0] * zero_len + [float("-inf")] * inf_len).expand(
                input.batch_seq_len_, input.batch_seq_len_).cuda(device)
        else:
            mask[idx] += torch.tensor([float("-inf")] * inf_len + [0] * zero_len).expand(
                input.batch_seq_len_, input.batch_seq_len_).cuda(device)

    return mask.to(torch.float32)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = einops.rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return einops.rearrange(x, "... d r -> ... (d r)")


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     angle: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    # data shape is: batch_size * max_seq_len * n_head * n_dim
    _, max_seq_len, _, dim_head = xq.shape

    cos = angle[0][:max_seq_len].view(max_seq_len, 1, dim_head)
    sin = angle[1][:max_seq_len].view(max_seq_len, 1, dim_head)

    xq = (xq * cos) + (rotate_half(xq) * sin)
    xk = (xk * cos) + (rotate_half(xk) * sin)
    return (xq, xk)


class RMSNorm():
    def __init__(self, weight: torch.Tensor, eps: float = 1e-06):
        self.norm_eps_ = eps
        self.weight_ = weight.to(torch.float32)

    def _norm(self, data: torch.Tensor) -> torch.Tensor:
        return data * torch.rsqrt(data.pow(2).mean(-1, keepdim=True) + self.norm_eps_)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self._norm(data.float()).type_as(data) * self.weight_


class Lora():
    def __init__(self, adapter_name: str):
        self.adapter_name_: str = adapter_name

        self.lora_a_: torch.Tensor = None
        self.lora_b_: torch.Tensor = None

        self.r_: int = 0
        self.alpha_: int = 0
        self.dropout_: float = 0.0
        self.scaling_: float = 0.0

    def set_parameter(self, r: int, alpha: int, dropout: float):
        self.r_ = r
        self.alpha_ = alpha
        self.dropout_ = dropout
        self.scaling_ = alpha / r

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data_ = F.dropout(data, self.dropout_)
        data_ @= self.lora_a_.transpose(0, 1)
        data_ @= self.lora_b_.transpose(0, 1)
        data_ *= self.scaling_
        return data_


class Linear():
    def __init__(self, weight: torch.Tensor, load_in_8bit: bool = True, device: str = None):
        if device is None:
            self.device_ = weight.device
        else:
            self.device_ = device

        if load_in_8bit:
            assert isinstance(
                weight, bitsandbytes.nn.Linear8bitLt), "error type."
        else:
            assert isinstance(weight, torch.nn.Linear), "error type."

        self.weight_ = weight
        self.enable_lora_: bool = False
        self.loras_: Dict[str, Lora] = {}

    def init_lora_weight(self, adapter_name: str, r: int, dim: int, alpha: int, dropout: float):
        if adapter_name not in self.loras_:
            self.loras_[adapter_name] = Lora(adapter_name)

        self.loras_[adapter_name].set_parameter(r, alpha, dropout)
        self.loras_[adapter_name].lora_a_ = torch.zeros(
            size=(r, dim), device=self.device_, requires_grad=True, dtype=torch.float32)
        self.loras_[adapter_name].lora_b_ = torch.zeros(
            size=(dim, r), device=self.device_, requires_grad=True, dtype=torch.float32)

        torch.nn.init.kaiming_normal_(
            self.loras_[adapter_name].lora_a_, a=math.sqrt(5))
        self.enable_lora_ = True

    def forward(self, data: torch.Tensor, input_args: MultiLoraBatchData) -> torch.Tensor:
        # data shape is: batch_size * max_seq_len * dim
        # result = data @ self.weight_.transpose(0, 1)
        result = self.weight_.forward(data)

        if not self.enable_lora_:
            return result

        for lora_config in input_args.lora_batch_data_config_:
            adapter_name = lora_config.adapter_name_
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            if adapter_name == "":
                continue

            result[start_idx: end_idx] += self.loras_[
                adapter_name].forward(data[start_idx:end_idx])

        return result


class Transformer():
    def __init__(self, layer_id: int, args: LlamaModelArgs):
        # attention
        self.wq_: Linear = None  # dim * dim
        self.wk_: Linear = None  # dim * dim
        self.wv_: Linear = None  # dim * dim
        self.wo_: Linear = None  # dim * dim
        # feed forward
        self.w1_: Linear = None  # also gate FNN * dim
        self.w2_: Linear = None  # also down dim * FNN
        self.w3_: Linear = None  # also up   FNN * dim
        # for lora linear
        # norm
        self.attention_norm_: RMSNorm = None  # dim
        self.ffn_norm_: RMSNorm = None        # dim
        # other arg
        self.layer_id_ = layer_id
        self.norm_eps_ = args.norm_eps_
        self.n_heads_ = args.n_heads_
        self.head_dim_ = args.dim_ // args.n_heads_

    def init_lora_layer_weight(self,
                               adapter_name: str,
                               r: int,
                               lora_alpha: int,
                               lora_dropout: float,
                               target: Dict[str, bool]):
        linear_layer_list = [self.wk_, self.wq_, self.wv_,
                             self.wo_, self.w1_, self.w2_, self.w3_]
        linear_layer_name_list = [
            "k_proj", "q_proj", "v_proj", "o_proj", "w1_proj", "w2_proj", "w3_proj"]

        for idx, layer_name in enumerate(linear_layer_name_list):
            if layer_name in target and target[layer_name]:
                linear_layer_list[idx].init_lora_weight(
                    adapter_name, r, self.head_dim_ * self.n_heads_, lora_alpha, lora_dropout)

    # @torch.compile
    def forward(self,
                data: torch.Tensor,
                mask: torch.Tensor,
                rope_angle: Tuple[torch.Tensor, torch.Tensor],
                input_args: MultiLoraBatchData):
        batch_size, max_seq_len, _ = data.shape

        attention_norm_data = self.attention_norm_.forward(data)

        xq = self.wq_.forward(attention_norm_data, input_args)
        xk = self.wk_.forward(attention_norm_data, input_args)
        xv = self.wv_.forward(attention_norm_data, input_args)

        # conver shape to multi head
        xq = xq.view(batch_size, max_seq_len, self.n_heads_, self.head_dim_)
        xk = xk.view(batch_size, max_seq_len, self.n_heads_, self.head_dim_)
        xv = xv.view(batch_size, max_seq_len, self.n_heads_, self.head_dim_)

        # apply rotary embedding
        xq, xk = apply_rotary_emb(xq, xk, rope_angle)

        # score shape is: batch_size * n_head * seq_len * dim_head
        # convert shape to: batch_size * seq_len * dim
        # attention_score = attention_score.transpose(
        #     1, 2).contiguous().view(batch_size, max_seq_len, -1)
        # attention_score = flash_attn_func(xq, xk, xv, causal=True)
        # attention_score = attention_score.view(batch_size, max_seq_len, -1)
        attention_score = xformers.ops.memory_efficient_attention(
            xq, xk, xv, mask)
        attention_score = attention_score.view(batch_size, max_seq_len, -1)

        # get output attention score
        data = data + self.wo_.forward(attention_score, input_args)

        # feed forward fully connected
        score_norm_data = self.ffn_norm_.forward(data)
        w1 = self.w1_.forward(score_norm_data, input_args)
        w3 = self.w3_.forward(score_norm_data, input_args)

        data = data + self.w2_.forward(F.silu(w1) * w3, input_args)

        return data


class LlamaModel():
    def __init__(self, args: LlamaModelArgs):
        # weight
        self.token_embedding_: torch.Tensor = None

        self.layers_: List[Transformer] = []
        for layer_id in range(args.n_layers_):
            self.layers_.append(Transformer(layer_id, args))

        self.norm_: RMSNorm = None          # dim
        self.output_: torch.Tensor = None   # vocab size * dim

        # cos and sin
        self.rope_angle_: Tuple[torch.Tensor, torch.Tensor] = precompute_rope_angle(
            args.dim_ // args.n_heads_, args.max_seq_len_, args.device)

        self.norm_eps_ = args.norm_eps_

        self.device_ = args.device
        self.n_heads_ = args.n_heads_
        self.vocab_size_ = args.vocab_size_
        self.pad_token_id_ = args.pad_token_id_
        self.dim_ = args.dim_

    def from_pretrained(path: str,
                        device: str,
                        load_in_8bit: bool = True):
        llama_model = LlamaForCausalLM.from_pretrained(
            path, load_in_8bit=load_in_8bit, device_map=device)

        llama_args = LlamaModelArgs()
        llama_args.dim_ = llama_model.config.hidden_size
        llama_args.n_heads_ = llama_model.config.num_attention_heads
        llama_args.n_layers_ = llama_model.config.num_hidden_layers
        llama_args.norm_eps_ = llama_model.config.rms_norm_eps
        llama_args.vocab_size_ = llama_model.config.vocab_size
        llama_args.max_seq_len_ = 4096 if hasattr(
            llama_model.config, "max_sequence_length") else llama_model.config.max_sequence_length
        llama_args.pad_token_id_ = -1
        llama_args.device = device

        model = LlamaModel(llama_args)

        model.token_embedding_ = llama_model.model.embed_tokens.weight.to(
            device=device)
        model.output_ = llama_model.lm_head.weight.to(
            dtype=torch.float32, device=device)
        model.norm_ = RMSNorm(llama_model.model.norm.weight.to(
            device=device), model.norm_eps_)

        for idx, layer in enumerate(llama_model.model.layers):
            model.layers_[idx].wq_ = Linear(
                layer.self_attn.q_proj, device=device)
            model.layers_[idx].wk_ = Linear(
                layer.self_attn.k_proj, device=device)
            model.layers_[idx].wv_ = Linear(
                layer.self_attn.v_proj, device=device)
            model.layers_[idx].wo_ = Linear(
                layer.self_attn.o_proj, device=device)
            model.layers_[idx].w1_ = Linear(layer.mlp.gate_proj, device=device)
            model.layers_[idx].w2_ = Linear(layer.mlp.down_proj, device=device)
            model.layers_[idx].w3_ = Linear(layer.mlp.up_proj, device=device)
            model.layers_[idx].attention_norm_ = RMSNorm(
                layer.input_layernorm.weight.to(device=device), model.norm_eps_)
            model.layers_[idx].ffn_norm_ = RMSNorm(
                layer.post_attention_layernorm.weight.to(device=device), model.norm_eps_)

        return model

    def forward(self, input: MultiLoraBatchData):
        tokens = torch.tensor(input.batch_tokens_,
                              dtype=torch.int).to(self.device_)
        data = F.embedding(tokens, self.token_embedding_,
                           padding_idx=self.pad_token_id_).requires_grad_(True)
        mask = precompute_mask(input, self.n_heads_, self.device_)

        def create_forward_for_checkpoint(module: Transformer):
            def forward_for_checkpoint(*inputs):
                return module.forward(*inputs)
            return forward_for_checkpoint

        for layer in self.layers_:
            data = torch.utils.checkpoint.checkpoint(
                create_forward_for_checkpoint(layer), data, mask, self.rope_angle_, input)

        data = self.norm_.forward(data)
        data @= self.output_.transpose(0, 1)

        return data

    def init_random_lora_weight(self, adapter_name: str,
                                r: int,
                                lora_alpha: int,
                                lora_dropout: float,
                                target: Dict[str, bool]):
        for transformer_layer in self.layers_:
            transformer_layer.init_lora_layer_weight(
                adapter_name, r, lora_alpha, lora_dropout, target)

    def get_train_paramas(self, config: Dict[str, str]) -> Dict[str, List[torch.Tensor]]:
        train_paramas = {}

        for transformer_layer in self.layers_:
            for lora_config in config["lora"]:
                adapter_name = lora_config["name"]
                if adapter_name not in train_paramas:
                    train_paramas[adapter_name] = []

                lora_layer_list = [transformer_layer.wq_.loras_, transformer_layer.wk_.loras_,
                                   transformer_layer.wv_.loras_, transformer_layer.wo_.loras_,
                                   transformer_layer.w1_.loras_, transformer_layer.w2_.loras_,
                                   transformer_layer.w3_.loras_]

                for lora_layer in lora_layer_list:
                    if adapter_name in lora_layer:
                        train_paramas[adapter_name].append(
                            lora_layer[adapter_name].lora_a_)
                        train_paramas[adapter_name].append(
                            lora_layer[adapter_name].lora_b_)

        return train_paramas
