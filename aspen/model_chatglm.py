from aspen.modelargs import LLMModelArgs, MultiLoraBatchData
from aspen.checkpoint import CheckpointRecomputeFunction
from aspen.model import LLMModel, RMSNorm, RotaryEmbedding
from aspen.model import apply_rotary_emb_to_one, repeat_kv, precompute_mask
from aspen.LoraLiner import Linear

import torch
import torch.nn.functional as F
import xformers.ops
import xformers.ops.fmha.attn_bias
from transformers import AutoModel
from typing import List, Dict, Optional


def swiglu(x: torch.Tensor) -> torch.Tensor:
    x = torch.chunk(x, 2, dim=-1)
    return F.silu(x[0]) * x[1]


class Transformer:
    def __init__(self, layer_id: int, args: LLMModelArgs):
        # attention
        self.query_key_value_: Linear = None
        self.attention_drop_out_: torch.nn.Dropout = None
        self.dense_: Linear = None
        # feed forward - MLP
        self.dense_h_to_4h_: Linear = None  # 4dim * dim
        self.dense_4h_to_h_: Linear = None  # dim * 4dim
        # norm
        self.input_layer_norm_: RMSNorm = None  # dim
        self.post_layer_norm_: RMSNorm = None   # dim
        # other args
        self.layer_id_ = layer_id

        self.n_heads_ = args.n_heads_
        self.n_kv_heads_ = args.n_kv_heads_
        self.n_rep_ = self.n_heads_ // self.n_kv_heads_
        self.head_dim_ = args.dim_ // args.n_heads_
        self.hidden_dropout_ = args.hidden_dropout_

    def init_lora_layer_weight(self,
                               adapter_name: str,
                               r: int,
                               lora_alpha: int,
                               lora_dropout: float,
                               target: Dict[str, bool],
                               weight: Optional[Dict[str, torch.Tensor]]):
        linear_layer_list = [self.query_key_value_,
                             self.dense_, self.dense_h_to_4h_, self.dense_4h_to_h_]
        linear_layer_name_list = [
            "qkv", "dense", "mlp_in", "mlp_out"]

        for idx, layer_name in enumerate(linear_layer_name_list):
            if layer_name in target and target[layer_name]:
                lora_a = None
                lora_b = None

                linear_layer_list[idx].init_lora_weight(
                    adapter_name, r, lora_alpha, lora_dropout, lora_a, lora_b)

    def forward(self,
                data: torch.Tensor,
                mask: torch.Tensor,
                rope_angle: torch.Tensor,
                input_args: MultiLoraBatchData):
        batch_size, seq_len, _ = data.shape

        layernorm_output = self.input_layer_norm_.forward(data)

        # batch_size, seq_len, dim => batch_size, seq_len, (qkv_hidden_size)
        # qkv_hidden_size = (qhead + khead + vhead) * head_dim
        mixed_x_layer = self.query_key_value_.forward(
            layernorm_output, input_args)
        assert (mixed_x_layer.shape[-1] == (self.n_heads_ +
                2 * self.n_kv_heads_) * self.head_dim_)

        # batch_size, seq_len, (num_head * head_dim)
        (query_layer, key_layer, value_layer) = mixed_x_layer.split(
            [self.n_heads_ * self.head_dim_,
             self.n_kv_heads_ * self.head_dim_,
             self.n_kv_heads_ * self.head_dim_],
            dim=-1
        )

        # => batch_size, seq_len, num_head, head_dim
        query_layer = query_layer.view(
            query_layer.size()[:-1] + (self.n_heads_, self.head_dim_))
        key_layer = key_layer.view(
            key_layer.size()[:-1] + (self.n_kv_heads_, self.head_dim_))
        value_layer = value_layer.view(
            value_layer.size()[:-1] + (self.n_kv_heads_, self.head_dim_))

        # batch_size, seq_len, num_head, head_dim
        query_layer = apply_rotary_emb_to_one(query_layer, rope_angle)
        key_layer = apply_rotary_emb_to_one(key_layer, rope_angle)

        # expand kv head to q head
        # the qkv is batch_size, seq_len, num_head, head_dim
        key_layer = repeat_kv(key_layer, self.n_rep_)
        value_layer = repeat_kv(value_layer, self.n_rep_)

        attention_score = xformers.ops.memory_efficient_attention(
            query_layer, key_layer, value_layer, mask)
        attention_score = attention_score.view(batch_size, seq_len, -1)

        attention_output = self.dense_.forward(attention_score, input_args)

        layernorm_input = F.dropout(
            attention_output, p=self.hidden_dropout_, training=not input_args.inference_model_)
        layernorm_input = data + layernorm_input

        # MLP
        layernorm_output = self.post_layer_norm_.forward(layernorm_input)

        h4 = self.dense_h_to_4h_.forward(layernorm_output, input_args)
        h4 = swiglu(h4)
        mlp_output = self.dense_4h_to_h_.forward(h4, input_args)

        mlp_output = F.dropout(
            mlp_output, p=self.hidden_dropout_, training=not input_args.inference_model_)
        mlp_output = mlp_output + layernorm_input

        return mlp_output


class ChatGLMModel(LLMModel):
    def __init__(self, args: LLMModelArgs):
        # weight
        self.token_embedding_: torch.Tensor = None

        self.layers_: List[Transformer] = []
        for layer_id in range(args.n_layers_):
            self.layers_.append(Transformer(layer_id, args))

        self.norm_: RMSNorm = None
        self.output_: torch.Tensor = None  # vocab size * dim

        self.rope_angle_ = RotaryEmbedding(
            args.dim_ // args.n_heads_ // 2, device=args.device, dtype=torch.float32)

        self.norm_eps_ = args.norm_eps_

        self.device_ = args.device
        self.pad_token_id_ = args.pad_token_id_
        self.n_heads_ = args.n_heads_
        self.vocab_size_ = args.vocab_size_
        self.dim_ = args.dim_

    def forward(self, input: MultiLoraBatchData):
        tokens = torch.tensor(input.batch_tokens_,
                              dtype=torch.int64).to(self.device_)

        # only for train
        def create_forward_for_checkpoint(module: Transformer):
            def forward_for_checkpoint(*inputs):
                return module.forward(*inputs)
            return forward_for_checkpoint

        # batch_size, seq_len, dim
        data = F.embedding(tokens, self.token_embedding_,
                           padding_idx=self.pad_token_id_).requires_grad_(True)
        mask = precompute_mask(input, self.n_heads_, self.device_, data.dtype)

        rope_angle = self.rope_angle_.forward(tokens.shape[1])
        for layer in self.layers_:
            if input.inference_model_:
                data = layer.forward(data, mask, rope_angle, input)
            else:
                data = CheckpointRecomputeFunction.apply(
                    create_forward_for_checkpoint(layer), data, mask, rope_angle, input)

        data = self.norm_.forward(data)
        data @= self.output_.transpose(0, 1)

        return data

    def from_pretrained(path: str,
                        device: str,
                        bits: int = None,
                        fp16: bool = True,
                        bf16: bool = True,
                        double_quant: bool = True,
                        quant_type: str = 'nf4',
                        log_fn=None):
        chatglm_model = AutoModel.from_pretrained(
            path,
            device_map=device,
            torch_dtype=torch.float32,
            quantization_bit=bits,
            trust_remote_code=True)

        # get config from chatglm config
        chatglm_args = LLMModelArgs()
        chatglm_args.device = device
        chatglm_args.norm_eps_ = chatglm_model.config.layernorm_epsilon
        chatglm_args.n_layers_ = chatglm_model.config.num_layers
        chatglm_args.pad_token_id_ = chatglm_model.config.pad_token_id
        chatglm_args.n_heads_ = chatglm_model.config.num_attention_heads
        chatglm_args.n_kv_heads_ = chatglm_model.config.multi_query_group_num
        chatglm_args.vocab_size_ = chatglm_model.config.vocab_size
        chatglm_args.dim_ = chatglm_model.config.hidden_size
        chatglm_args.hidden_dropout_ = chatglm_model.config.hidden_dropout

        model = ChatGLMModel(chatglm_args)

        # get the weight from transformer
        model.token_embedding_ = chatglm_model.transformer.embedding.word_embeddings.weight.to(
            device).requires_grad_(False)
        model.output_ = chatglm_model.transformer.output_layer.weight.to(
            device=device).requires_grad_(False)
        model.norm_ = RMSNorm(chatglm_model.transformer.encoder.final_layernorm.weight.to(
            device=device).requires_grad_(False), model.norm_eps_)

        for idx, layer in enumerate(chatglm_model.transformer.encoder.layers):
            # linear
            model.layers_[idx].query_key_value_ = Linear(
                layer.self_attention.query_key_value.requires_grad_(False), device=device)
            model.layers_[idx].dense_ = Linear(
                layer.self_attention.dense.requires_grad_(False), device=device)
            model.layers_[idx].attention_drop_out_ = torch.nn.Dropout(
                chatglm_model.config.attention_dropout)
            # mlp
            model.layers_[idx].dense_h_to_4h_ = Linear(
                layer.mlp.dense_h_to_4h.requires_grad_(False), device=device)
            model.layers_[idx].dense_4h_to_h_ = Linear(
                layer.mlp.dense_4h_to_h.requires_grad_(False), device=device)
            # norm
            model.layers_[idx].input_layer_norm_ = RMSNorm(
                layer.input_layernorm.weight.to(device=device).requires_grad_(False), model.norm_eps_)
            model.layers_[idx].post_layer_norm_ = RMSNorm(
                layer.post_attention_layernorm.weight.to(device=device).requires_grad_(False), model.norm_eps_)

        return model

    def init_lora_weight(self, adapter_name: str,
                         r: int,
                         lora_alpha: int,
                         lora_dropout: float,
                         target: Dict[str, bool],
                         weight: Optional[Dict[str, torch.Tensor]]):
        for transformer_layer in self.layers_:
            transformer_layer.init_lora_layer_weight(
                adapter_name, r, lora_alpha, lora_dropout, target, weight)

    def get_train_paramas(self, config: Dict[str, str]) -> Dict[str, List[torch.Tensor]]:
        train_paramas = {}

        for transformer_layer in self.layers_:
            for lora_config in config["lora"]:
                adapter_name = lora_config["name"]
                if adapter_name not in train_paramas:
                    train_paramas[adapter_name] = []

                lora_layer_list = [transformer_layer.query_key_value_.loras_,
                                   transformer_layer.dense_.loras_,
                                   transformer_layer.dense_h_to_4h_.loras_,
                                   transformer_layer.dense_4h_to_h_.loras_]

                for lora_layer in lora_layer_list:
                    if adapter_name in lora_layer:
                        train_paramas[adapter_name].append(
                            lora_layer[adapter_name].lora_a_)
                        train_paramas[adapter_name].append(
                            lora_layer[adapter_name].lora_b_)

        return train_paramas
