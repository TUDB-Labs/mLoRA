from mlora.modelargs import LLMModelArgs, LLMModelOutput, MultiLoraBatchData
from mlora.modelargs import Masks, LoraConfig, MixConfig, lora_config_factory
from mlora.attention import LlamaAttention, llama_attention_factory
from mlora.checkpoint import CheckpointRecomputeFunction
from mlora.model import LLMOutput, LLMModel
from mlora.feed_forward import FeedForward
from mlora.lora_linear import Linear
from mlora.generate import GenerateConfig
from mlora.mix_lora import router_loss_factory
from mlora.tasks import SequenceClassificationTask, task_dict
from mlora.backends import get_backend
from mlora.utils import _is_package_available

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from typing import List, Dict, Tuple, Optional
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM
from collections import OrderedDict
import logging
import json
import math
import os

if _is_package_available("bitsandbytes"):
    from transformers import BitsAndBytesConfig
else:
    from mlora.utils import BitsAndBytesConfig


# input_tokens shape is: batch_size * seq_len
#   default: upper triangular matrix like below, i.e. diagonal = 1
#            0 -inf -inf
#            0    0 -inf
#            0    0    0
# additional_mask: batch_size * seq_len
#   default: is None the matrix like default, if set true, the mask metric will be -inf
#   example: [[True, False, False]]
#           -inf -inf -inf
#           -inf    0 -inf
#           -inf    0    0
def precompute_mask(input_tokens: torch.Tensor,
                    n_heads: int,
                    device: str,
                    additional_mask: List[Masks] = None,
                    diagonal: int = 1,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    batch_size, seq_len = input_tokens.shape

    TORCH_MIN_VALUE = torch.finfo(torch.float32).min
    mask = torch.full((batch_size, n_heads, seq_len, seq_len),
                      TORCH_MIN_VALUE, device=device, dtype=torch.float32)
    mask = torch.triu(mask, diagonal=diagonal)

    if additional_mask is not None:
        masks_metric = ~torch.tensor(
            additional_mask, dtype=torch.bool, device=device)
        masks_metric = masks_metric.view(batch_size, 1, 1, seq_len)
        masks_metric = masks_metric.expand(-1, n_heads, seq_len, -1)
        mask = torch.masked_fill(mask, masks_metric, TORCH_MIN_VALUE)

    mask.requires_grad_(False)

    return mask.to(device=device, dtype=dtype)


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.norm_eps_ = eps
        self.weight_ = weight

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        input_dtype = data.dtype
        v = data.to(torch.float32).pow(2).mean(-1, keepdim=True)
        data = data * torch.rsqrt(v + self.norm_eps_)

        return (self.weight_ * data).to(input_dtype)


class LlamaEmbedding(torch.nn.Module):
    def __init__(self, embedding: torch.Tensor, pad_token: int):
        super().__init__()
        self.token_embedding_: torch.Tensor = embedding
        self.padding_idx_: int = pad_token

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        data = F.embedding(tokens, self.token_embedding_,
                           padding_idx=self.padding_idx_)
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


class LlamaOutputLayer(torch.nn.Module):
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


class LlamaDecoderLayer(torch.nn.Module):
    def __init__(self, layer_idx: int, args: LLMModelArgs):
        super().__init__()
        # attention
        self.attn_norm_: LlamaRMSNorm = None  # dim
        self.attn_: LlamaAttention = None
        # feed forward
        self.ffn_norm_: LlamaRMSNorm = None  # dim
        self.ffn_: FeedForward = None
        # other arg
        self.layer_id_ = layer_idx
        self.dim_ = args.dim_
        self.norm_eps_ = args.norm_eps_
        self.dtype_ = args.dtype_

    def init_lora_layer_weight(self, config: LoraConfig, weight: Optional[Dict[str, torch.Tensor]]):
        target = config.target_modules_
        linear_layer_list = [self.attn_.wk_, self.attn_.wq_, self.attn_.wv_,
                             self.attn_.wo_, self.ffn_.w1_, self.ffn_.w2_, self.ffn_.w3_]
        linear_layer_name_list = [
            "k_proj", "q_proj", "v_proj", "o_proj", "w1_proj", "w2_proj", "w3_proj"]

        if isinstance(config, MixConfig):
            # Inject LoRA configs into FFN layer
            gate_layer_name = f"mixlora.layers.{self.layer_id_}.gate.weight"
            self.ffn_.init_moe_weight(in_features=self.dim_, config=config,
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
                            config.expert_config(expert_idx), (lora_a, lora_b))
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
                        config, (lora_a, lora_b))

    def forward(self,
                hidden_states: torch.Tensor,
                input_args: MultiLoraBatchData,
                attention_mask: Optional[torch.Tensor] = None,
                router_logits: Optional[List[List]] = None):
        # Self Attention
        residual = hidden_states
        hidden_states = self.attn_norm_(hidden_states)
        hidden_states = self.attn_(hidden_states, input_args, attention_mask)
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm_(hidden_states)
        hidden_states = self.ffn_(hidden_states, input_args, router_logits)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaSequentialWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.wrapper_module_ = module

    def name(self) -> str:
        return type(self.wrapper_module_).__name__

    def forward(self, input: Tuple) -> Tuple:
        module_name = self.name()

        if module_name == "LlamaEmbedding":
            output = self.wrapper_module_.forward(input[0])
            if input[-1]:
                output = output.requires_grad_(True)
            return (output, ) + input[1:]
        elif module_name == "LlamaRMSNorm":
            output = self.wrapper_module_.forward(input[0])
            return (output, ) + input[1:]
        elif module_name == "LlamaOutputLayer":
            output = self.wrapper_module_.forward(*input[:2])
            return (output, ) + input[1:]
        elif module_name == "LlamaDecoderLayer":
            if input[-1]:
                output = CheckpointRecomputeFunction.apply(
                    self.wrapper_module_.forward, *input[:-1])
            else:
                output = self.wrapper_module_.forward(*input[:-1])
            return (output, ) + input[1:]
        else:
            raise f"module invalid: {module_name}"


LlamaCompatibleModelTypes = ["mistral", "qwen2"]


class LlamaModel(LLMModel):
    def __init__(self, args: LLMModelArgs):
        self.name_or_path_ = args.name_or_path_
        # weight
        self.token_embedding_: LlamaEmbedding = None

        self.layers_: List[LlamaDecoderLayer] = []
        for layer_id in range(args.n_layers_):
            self.layers_.append(LlamaDecoderLayer(layer_id, args))

        self.norm_: LlamaRMSNorm = None
        self.lm_head_ = torch.nn.Linear(
            args.dim_, args.vocab_size_, bias=False, device=args.device_, dtype=args.dtype_)
        self.output_: LlamaOutputLayer = LlamaOutputLayer()
        self.seq_module_: torch.nn.Sequential = None

        if args.vocab_size_ >= torch.finfo(args.dtype_).max:
            logging.warn(
                f"vocab_size >= max({args.dtype_}), consider load model with higher precision.")

        # configs
        self.attn_implementation_ = args.attn_implementation_
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
                              dtype=torch.int64, device=self.device_)

        # prepare mask
        if self.attn_implementation_ == "flash_attn":
            if input.attention_masks_ is not None and 0 in input.attention_masks_:
                # 2d mask is passed through the layers
                mask = torch.tensor(input.attention_masks_,
                                    dtype=torch.int64, device=self.device_)
            else:
                mask = None
        else:
            mask = precompute_mask(
                input_tokens=tokens,
                n_heads=1 if input.inference_mode_ else self.n_heads_,
                device=self.device_,
                additional_mask=input.attention_masks_,
                diagonal=input.inference_seq_pos_ + 1 if input.inference_mode_ else 1,
                dtype=self.dtype_)

        # routing data
        router_logits: List[List] = list(
            [] for _ in range(len(input.lora_batch_data_config_)))

        if input.inference_mode_:
            input.gradient_checkpoint_ = False

        data = (tokens, input, mask,
                router_logits, input.gradient_checkpoint_)

        for seq_layer in self.seq_module_:
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
            output_data.aux_loss = loss_fn(router_logits[idx])

        return output

    def init_lora_layer_weight(self, config: LoraConfig, weight: Optional[Dict[str, torch.Tensor]]):
        self.adapter_configs_[config.adapter_name] = config
        # init output layer
        if config.task_name in task_dict and isinstance(task_dict[config.task_name], SequenceClassificationTask):
            output_layer = ClassificationOutputLayer(
                **task_dict[config.task_name].init_kwargs(),
                hidden_size=self.dim_,
                pad_token_id=self.pad_token_id_,
                device=self.device_,
                weight=weight)
        else:
            output_layer = CasualOutputLayer(
                vocab_size=self.vocab_size_,
                weight=self.lm_head_)

        self.output_.layers_[config.adapter_name] = output_layer
        if config.adapter_name == "default" and weight is None:
            return
        # init transformer layers
        for transformer_layer in self.layers_:
            transformer_layer.init_lora_layer_weight(config, weight)

    def from_pretrained(path: str,
                        device: str,
                        bits: int = None,
                        attn_impl: str = "eager",
                        use_sliding_window: bool = False,
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
        if not get_backend().is_bf16_supported():
            if load_dtype == torch.bfloat16:
                logging.warning("bf16 is not available. deprecated to fp16.")
                load_dtype = torch.float16

            if bits in [4, 8] and compute_dtype == torch.bfloat16:
                logging.warning("bf16 is not available. deprecated to fp16.")
                compute_dtype = torch.float16

        if bits in [4, 8]:
            logging.info(f"Loading model with quantization, bits = {bits}.")
            llama_model = AutoModelForCausalLM.from_pretrained(
                path,
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
            llama_model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map=device,
                torch_dtype=load_dtype)

        if llama_model.config.model_type != "llama":
            if llama_model.config.model_type in LlamaCompatibleModelTypes:
                logging.info(
                    f"Loading {llama_model.config.model_type} model with llama compatible mode.")
            else:
                logging.warning(
                    f"Unsupported model type {llama_model.config.model_type}, loading with llama compatible mode.")

        llama_model.requires_grad_(False)

        llama_args = LLMModelArgs()
        llama_args.attn_implementation_ = attn_impl
        logging.info(f"Use {attn_impl} as attention implementation.")
        llama_args.name_or_path_ = llama_model.config.name_or_path
        llama_args.dim_ = llama_model.config.hidden_size
        llama_args.n_heads_ = llama_model.config.num_attention_heads
        llama_args.n_kv_heads_ = llama_args.n_heads_ if not hasattr(
            llama_model.config, "num_key_value_heads") else llama_model.config.num_key_value_heads
        llama_args.n_layers_ = llama_model.config.num_hidden_layers
        llama_args.norm_eps_ = llama_model.config.rms_norm_eps
        llama_args.vocab_size_ = llama_model.config.vocab_size
        llama_args.max_seq_len_ = 4096 if not use_sliding_window and not hasattr(
            llama_model.config, "max_position_embeddings") else llama_model.config.max_position_embeddings
        if hasattr(llama_model.config, "use_sliding_window"):
            llama_args.use_sliding_window_ = use_sliding_window or llama_model.config.use_sliding_window
        else:
            llama_args.use_sliding_window_ = use_sliding_window
        if hasattr(llama_model.config, "max_window_layers"):
            llama_args.max_window_layers_ = llama_model.config.max_window_layers
        if hasattr(llama_model.config, "sliding_window"):
            llama_args.sliding_window_ = llama_model.config.sliding_window
        llama_args.pad_token_id_ = llama_model.config.pad_token_id
        llama_args.rope_theta_ = llama_model.config.rope_theta
        if llama_args.pad_token_id_ is None:
            llama_args.pad_token_id_ = -1
        llama_args.dtype_ = llama_model.dtype

        llama_args.device_ = device

        model = LlamaModel(llama_args)

        model.token_embedding_ = LlamaEmbedding(
            llama_model.model.embed_tokens.weight, llama_args.pad_token_id_)

        model.norm_ = LlamaRMSNorm(
            llama_model.model.norm.weight, model.norm_eps_)

        with torch.no_grad():
            model.lm_head_.weight.copy_(llama_model.lm_head.weight)

        for idx, layer in enumerate(llama_model.model.layers):
            model.layers_[idx].attn_norm_ = LlamaRMSNorm(
                layer.input_layernorm.weight, model.norm_eps_)
            model.layers_[idx].attn_ = llama_attention_factory(
                model_type=llama_model.config.model_type,
                wq=Linear(layer.self_attn.q_proj, device=device),
                wk=Linear(layer.self_attn.k_proj, device=device),
                wv=Linear(layer.self_attn.v_proj, device=device),
                wo=Linear(layer.self_attn.o_proj, device=device),
                args=llama_args,
                layer_idx=idx
            )
            model.layers_[idx].ffn_norm_ = LlamaRMSNorm(
                layer.post_attention_layernorm.weight, model.norm_eps_)
            model.layers_[idx].ffn_ = FeedForward(
                w1=Linear(layer.mlp.gate_proj, device=device),
                w2=Linear(layer.mlp.down_proj, device=device),
                w3=Linear(layer.mlp.up_proj, device=device),
                args=llama_args
            )

        # convert to sequential module for use
        model.seq_module_ = model.sequential_module()

        return model

    def get_generate_paramas(self) -> Dict[str, GenerateConfig]:
        generate_paramas = {}
        for adapter_name in self.adapter_configs_.keys():
            generate_paramas[adapter_name] = GenerateConfig(
                adapter_name=adapter_name)
        return generate_paramas

    def get_lora_weight_dict(self, lora_name: str) -> Dict[str, torch.Tensor]:
        # return the lora weight and target_module's name
        lora_weight_dict = self.output_.layers_[lora_name].state_dict()
        for idx, transformer_layer in enumerate(self.layers_):
            if isinstance(self.adapter_configs_[lora_name], MixConfig):
                layer_prefix_name = f"mixlora.layers.{idx}.self_attn."
            else:
                layer_prefix_name = f"base_model.model.model.layers.{idx}.self_attn."

            lora_layer_list = [transformer_layer.attn_.wq_, transformer_layer.attn_.wk_,
                               transformer_layer.attn_.wv_, transformer_layer.attn_.wo_,
                               transformer_layer.ffn_.w1_, transformer_layer.ffn_.w2_,
                               transformer_layer.ffn_.w3_]
            lora_layer_name_list = [
                "q_proj", "k_proj", "v_proj", "o_proj", "w1_proj", "w2_proj", "w3_proj"]
            for idx, lora_layer in enumerate(lora_layer_list):
                if lora_name in lora_layer.loras_:
                    prefix_name = layer_prefix_name + lora_layer_name_list[idx]
                    lora_weight_dict[f"{prefix_name}.lora_A.weight"] = lora_layer.loras_[
                        lora_name].lora_a_.weight
                    lora_weight_dict[f"{prefix_name}.lora_B.weight"] = lora_layer.loras_[
                        lora_name].lora_b_.weight
                elif lora_name in transformer_layer.ffn_.moes_:
                    moe_layer_prefix_name = f"mixlora.layers.{transformer_layer.layer_id_}."
                    for expert_idx in range(transformer_layer.ffn_.moes_[lora_name].experts_):
                        moe_lora_name = f"moe.{lora_name}.experts.{expert_idx}"
                        if moe_lora_name in lora_layer.loras_:
                            lora_weight_dict[
                                moe_layer_prefix_name
                                + f"experts.{expert_idx}."
                                + f"{lora_layer_name_list[idx]}.lora_A.weight"
                            ] = lora_layer.loras_[moe_lora_name].lora_a_.weight
                            lora_weight_dict[
                                moe_layer_prefix_name
                                + f"experts.{expert_idx}."
                                + f"{lora_layer_name_list[idx]}.lora_B.weight"
                            ] = lora_layer.loras_[moe_lora_name].lora_b_.weight

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
            lora_config.adapter_name = adapter_name
            lora_weight = torch.load(
                path + os.sep + "adapter_model.bin", map_location=self.device_)
        else:
            lora_config = LoraConfig(adapter_name=path)
            lora_weight = None
        self.init_lora_layer_weight(lora_config, lora_weight)
        return adapter_name
