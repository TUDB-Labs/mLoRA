from mlora.modelargs import LLMModelArgs, LLMModelOutput, MultiLoraBatchData
from mlora.modelargs import LoraConfig, MixConfig, lora_config_factory
from mlora.checkpoint import CheckpointRecomputeFunction
from mlora.model import repeat_kv, apply_rotary_emb, precompute_rope_angle, precompute_mask
from mlora.model import RMSNorm, LLMOutput, LLMModel
from mlora.feed_forward import FeedForward
from mlora.lora_liner import Linear
from mlora.generate import GenerateConfig
from mlora.mix_lora import router_loss_factory
from mlora.tasks import classification_tasks

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import xformers.ops
import xformers.ops.fmha.attn_bias
from typing import List, Dict, Tuple, Optional
from huggingface_hub import snapshot_download
from transformers import BitsAndBytesConfig
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


class CasualOutputLayer(LLMOutput):
    def __init__(self, vocab_size: int, weight: torch.nn.Linear):
        super().__init__()
        self.vocab_size_: int = vocab_size
        self.lm_head_: torch.nn.Linear = weight

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.lm_head_(data).float()

    def loss(self,
             input_ids: torch.Tensor,
             output_logits: torch.Tensor,
             labels: List[List[int]]) -> torch.Tensor:
        labels = torch.tensor(labels, dtype=torch.long,
                              device=output_logits.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(output_logits[..., :-1, :].contiguous().view(-1, self.vocab_size_),
                       labels[..., 1:].contiguous().view(-1))


class ClassificationOutputLayer(LLMOutput):
    def __init__(self,
                 task_type: str,
                 num_labels: int,
                 label_dtype: torch.dtype,
                 hidden_size: int,
                 pad_token_id: int,
                 device: str,
                 weight: Optional[torch.Tensor]):
        super().__init__()
        self.label_dtype_ = label_dtype
        self.num_labels_ = num_labels
        self.task_type_ = task_type
        self.pad_id_ = pad_token_id
        self.score_ = torch.nn.Linear(
            hidden_size, self.num_labels_, bias=False, dtype=torch.float32, device=device)
        if weight is None:
            torch.nn.init.kaiming_normal_(
                self.score_.weight, a=math.sqrt(5))
        else:
            with torch.no_grad():
                self.score_.weight.copy_(weight["classifier"])

    def state_dict(self):
        return {"classifier": self.score_.weight}

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.score_(data.to(torch.float32))

    def loss(self,
             input_ids: torch.Tensor,
             output_logits: torch.Tensor,
             labels: List[List[int]]) -> torch.Tensor:
        labels = torch.tensor(
            labels, dtype=self.label_dtype_, device=output_logits.device)
        batch_size = input_ids.shape[0]
        sequence_lengths = (torch.eq(
            input_ids, self.pad_id_).int().argmax(-1) - 1).to(output_logits.device)
        pooled_logits = output_logits[torch.arange(
            batch_size, device=output_logits.device), sequence_lengths]
        if self.task_type_ == "single_label_classification":
            loss_fn = torch.nn.CrossEntropyLoss()
            return loss_fn(pooled_logits.view(-1, self.num_labels_), labels.view(-1))
        elif self.task_type_ == "multi_label_classification":
            loss_fn = torch.nn.BCEWithLogitsLoss()
            return loss_fn(pooled_logits, labels)
        else:
            raise ValueError(f"unknown task type {self.task_type_}")


class OutputLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_: torch.ModuleDict = {}

    def forward(self, data: torch.Tensor,
                input_args: MultiLoraBatchData) -> List[LLMModelOutput]:
        outputs = []
        for lora_config in input_args.lora_batch_data_config_:
            adapter_name = lora_config.adapter_name_
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_

            assert adapter_name != "" and adapter_name in self.layers_
            layer = self.layers_[adapter_name]
            outputs.append(LLMModelOutput(adapter_name=adapter_name,
                                          logits=layer.forward(
                                              data[start_idx:end_idx]),
                                          loss_fn_=layer.loss))

        return outputs


class Transformer(torch.nn.Module):
    def __init__(self, layer_id: int, args: LLMModelArgs):
        super().__init__()
        # cos and sin
        self.rope_angle_: Tuple[torch.Tensor, torch.Tensor] = precompute_rope_angle(
            args.dim_ // args.n_heads_, args.max_seq_len_, args.rope_theta_, args.device_)
        # attention
        self.attention_norm_: RMSNorm = None  # dim
        self.attention_cos_: torch.Tensor = None
        self.attention_sin_: torch.Tensor = None
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
        self.dtype_ = args.dtype_

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
                input_args: MultiLoraBatchData,
                mask: torch.Tensor,
                router_logits: List[List] = None):
        batch_size, max_seq_len, _ = data.shape

        attention_norm_data = self.attention_norm_.forward(data)

        xq = self.wq_.forward(attention_norm_data, input_args)
        xk = self.wk_.forward(attention_norm_data, input_args)
        xv = self.wv_.forward(attention_norm_data, input_args)

        # conver shape to multi head
        xq = xq.view(batch_size, max_seq_len, self.n_heads_,
                     self.head_dim_).transpose(1, 2)
        xk = xk.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)
        xv = xv.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)

        # apply rotary embedding
        cos = self.rope_angle_[0][:max_seq_len].to(xv.dtype)
        sin = self.rope_angle_[1][:max_seq_len].to(xv.dtype)
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        # for llama2 need to repeat the heads
        # before dim: batch_size, n_kv_head, seq_len, head_dim
        # after dim: batch_size, n_head, seq_len, head_dim
        xk = repeat_kv(xk, self.n_rep_)
        xv = repeat_kv(xv, self.n_rep_)

        if input_args.inference_mode_:
            attention_score = torch.matmul(
                xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim_)
            if mask is not None:
                attention_score = attention_score + mask
            attention_score = F.softmax(
                attention_score.float(), dim=-1).to(xq.dtype)
            attention_score = torch.matmul(attention_score, xv)
            attention_score = attention_score.transpose(1, 2).contiguous()
        else:
            xq = xq.transpose(1, 2)
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)
            attention_score = xformers.ops.memory_efficient_attention(
                xq, xk, xv, mask)

        attention_score = attention_score.reshape(batch_size, max_seq_len, -1)

        # get output attention score
        data = data + self.wo_.forward(attention_score, input_args)

        # feed forward fully connected
        data = data + self.ffn_.forward(data, input_args, router_logits)

        return data


class LlamaSequentialWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.wrapper_module_ = module

    def name(self) -> str:
        return type(self.wrapper_module_).__name__

    def forward(self, input: Tuple) -> Tuple:
        module_name = self.name()

        if module_name == "Embedding" or module_name == "RMSNorm":
            output = self.wrapper_module_.forward(input[0])
            return (output, ) + input[1:]
        elif module_name == "OutputLayer":
            output = self.wrapper_module_.forward(*input[:2])
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

        self.norm_: RMSNorm = None    # dim
        self.lm_head_ = torch.nn.Linear(
            args.dim_, args.vocab_size_, bias=False, device=args.device_, dtype=args.dtype_)
        self.output_: OutputLayer = OutputLayer()

        self.norm_eps_ = args.norm_eps_

        self.device_ = args.device_
        self.n_heads_ = args.n_heads_
        self.n_kv_heads_ = args.n_kv_heads_
        self.vocab_size_ = args.vocab_size_
        self.pad_token_id_ = args.pad_token_id_
        self.max_seq_len_ = args.max_seq_len_
        self.dim_ = args.dim_
        self.dtype_ = args.dtype_

        # adapter configs
        self.adapter_configs_: Dict[str, LoraConfig] = {}

    # train model or inference model: output is probs
    def forward(self, input: MultiLoraBatchData,
                labels: List[List[int]] = None) -> List[LLMModelOutput]:
        tokens = torch.tensor(input.batch_tokens_,
                              dtype=torch.long, device=self.device_)

        seq_module = self.sequential_module()

        # prepare mask
        attention_masks = precompute_mask(tokens, input.attention_masks_,
                                          1 if input.inference_mode_ else self.n_heads_,
                                          input.inference_seq_pos_ if input.inference_mode_ else 0,
                                          self.device_, self.dtype_)

        # routing data
        router_logits: List[List] = list(
            [] for _ in range(len(input.lora_batch_data_config_)))

        if input.inference_mode_:
            input.gradient_checkpoint_ = False

        data = (tokens, input, attention_masks,
                router_logits, input.gradient_checkpoint_)

        for seq_layer in seq_module:
            data = seq_layer.forward(data)

        output = data[0]
        assert isinstance(output, List)
        for idx, lora_config in enumerate(input.lora_batch_data_config_):
            output_data = output[idx]
            assert isinstance(output_data, LLMModelOutput)
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_
            output_data.batch_start_idx_ = start_idx
            output_data.batch_end_idx_ = end_idx
            if labels is None:
                continue
            # compute loss when labels provided
            output_data.loss = output_data.loss_fn_(
                tokens[start_idx:end_idx], output_data.logits, labels[start_idx:end_idx])
            output_data.loss_fn_ = None
            if len(router_logits[idx]) == 0:
                continue
            # compute router loss when router logits is available
            loss_fn = router_loss_factory(
                self.adapter_configs_[output_data.adapter_name])
            output_data.loss += loss_fn(router_logits[idx])

        return output

    def init_lora_layer_weight(self, config: LoraConfig, weight: Optional[Dict[str, torch.Tensor]]):
        self.adapter_configs_[config.adapter_name_] = config
        # init output layer
        if config.task_type_ == "casual":
            output_layer = CasualOutputLayer(
                vocab_size=self.vocab_size_,
                weight=self.lm_head_)
        else:
            output_layer = ClassificationOutputLayer(
                **classification_tasks[config.task_type_].init_kwargs(),
                hidden_size=self.dim_,
                pad_token_id=self.pad_token_id_,
                device=self.device_,
                weight=weight)
        self.output_.layers_[config.adapter_name_] = output_layer
        if config.adapter_name_ == "default":
            return
        # init transformer layers
        for transformer_layer in self.layers_:
            transformer_layer.init_lora_layer_weight(config, weight)

    def from_pretrained(path: str,
                        device: str,
                        bits: int = None,
                        load_dtype: torch.dtype = torch.bfloat16,
                        compute_dtype: torch.dtype = torch.bfloat16,
                        double_quant: bool = True,
                        quant_type: str = 'nf4',
                        ) -> LLMModel:
        # load_dtype will change the precision of LLaMA pre-trained model
        # when loading with quantization (bits = 8 or bits = 4), load_dtype will only influence the actual computing precision
        if load_dtype not in [torch.bfloat16, torch.float16, torch.float32]:
            raise ValueError(f"unsupported load dtype {load_dtype}")

        if compute_dtype not in [torch.bfloat16, torch.float16, torch.float32]:
            raise ValueError(f"unsupported compute dtype {compute_dtype}")

        if load_dtype in [torch.bfloat16, torch.float16]:
            logging.info("Loading model with half precision.")

        # BFloat16 is only supported after Ampere GPUs
        if not torch.cuda.is_bf16_supported():
            if load_dtype == torch.bfloat16:
                logging.warning("bf16 is not available. deprecated to fp16.")
                load_dtype = torch.float16

            if compute_dtype == torch.bfloat16:
                logging.warning("bf16 is not available. deprecated to fp16.")
                compute_dtype = torch.float16

        if bits in [4, 8]:
            logging.info(f"Loading model with quantization, bits = {bits}.")
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
                torch_dtype=load_dtype)
        else:
            llama_model = LlamaForCausalLM.from_pretrained(
                path,
                device_map=device,
                torch_dtype=load_dtype)

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
        llama_args.rope_theta_ = llama_model.config.rope_theta
        if llama_args.pad_token_id_ is None:
            llama_args.pad_token_id_ = -1
        llama_args.dtype_ = llama_model.dtype

        llama_args.device_ = device

        model = LlamaModel(llama_args)

        embedding_weight = llama_model.model.embed_tokens.weight.to(
            device=device).detach()
        model.token_embedding_ = Embedding(
            embedding_weight, llama_args.pad_token_id_)

        with torch.no_grad():
            model.lm_head_.weight.copy_(llama_model.lm_head.weight)
        model.lm_head_.requires_grad_(False)

        norm_weight = llama_model.model.norm.weight.to(
            device=device).detach()
        model.norm_ = RMSNorm(norm_weight, model.norm_eps_)

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
                    device=device).detach(), model.norm_eps_),
                w1=Linear(layer.mlp.gate_proj, device=device),
                w2=Linear(layer.mlp.down_proj, device=device),
                w3=Linear(layer.mlp.up_proj, device=device),
                device=device
            )
            model.layers_[idx].attention_norm_ = RMSNorm(
                layer.input_layernorm.weight.to(device=device).detach(), model.norm_eps_)

        return model

    def get_train_paramas(self) -> Dict[str, List[torch.Tensor]]:
        train_paramas = {}

        for name, layer in self.output_.layers_.items():
            train_paramas[name] = list(layer.state_dict().values())

        for transformer_layer in self.layers_:
            for adapter_name, lora_config in self.adapter_configs_.items():
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

    def get_lora_weight_dict(self, lora_name: str) -> Dict[str, torch.Tensor]:
        # return the lora weight and target_module's name
        lora_weight_dict = self.output_.layers_[lora_name].state_dict()
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
        if path != "default":
            if not os.path.exists(path):
                path = snapshot_download(repo_id=path, repo_type="model")
            with open(path + os.sep + "adapter_config.json", 'r', encoding='utf8') as fp:
                lora_config = lora_config_factory(json.load(fp))
            lora_config.adapter_name_ = adapter_name
            lora_weight = torch.load(
                path + os.sep + "adapter_model.bin", map_location=self.device_)
        else:
            lora_config = LoraConfig(adapter_name_=path)
            lora_weight = None
        self.init_lora_layer_weight(lora_config, lora_weight)
        return adapter_name
