from mlora.common import (
    LLMOutput,
    LLMDecoder,
    LLMForCausalLM,
    LLMModelArgs,
    LLMModelOutput,
    MultiLoraBatchData,
    LoraConfig,
    MixConfig,
    router_loss_factory,
    lora_config_factory,
)
from mlora.tasks import SequenceClassificationTask, task_dict
from mlora.utils import is_package_available
from mlora.models import from_pretrained
from mlora.backends import get_backend

from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM
from typing import Tuple, List, Dict, Optional

import logging
import torch
import json
import math
import os

if is_package_available("bitsandbytes"):
    from transformers import BitsAndBytesConfig
else:
    from mlora.utils import BitsAndBytesConfig


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
             labels) -> torch.Tensor:
        if isinstance(labels, torch.Tensor):
            labels = labels.clone().detach().to(dtype=torch.long,
                                                device=output_logits.device)
        else:
            labels = torch.tensor(
                labels, dtype=torch.long, device=output_logits.device)
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
             labels) -> torch.Tensor:
        if isinstance(labels, torch.Tensor):
            labels = labels.clone().detach().to(dtype=self.label_dtype_,
                                                device=output_logits.device)
        else:
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


def init_lora_layer_weight(
        layer: LLMDecoder, args: LLMModelArgs, config: LoraConfig, weight: Optional[Dict[str, torch.Tensor]]):
    target = config.target_modules_
    linear_layer_list = []
    linear_layer_name_list = []
    for name, module in layer.state_dict().items():
        linear_layer_list.append(module)
        linear_layer_name_list.append(name)

    if isinstance(config, MixConfig):
        if config.sparse_step_ is None or layer.layer_id_ % config.sparse_step_ == 1:
            # Inject LoRA configs into FFN layer
            gate_layer_name = f"mixlora.layers.{layer.layer_id_}.gate.weight"
            layer.mlp_.init_moe_weight(args=args, config=config,
                                       gate=None if weight is None else weight[gate_layer_name])
            moe_layer_name_list = list(layer.mlp_.state_dict().keys())
        else:
            moe_layer_name_list = []

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
                        lora_a_name = f"mixlora.layers.{layer.layer_id_}.experts.{expert_idx}.{layer_name}.lora_A.weight"
                        lora_b_name = f"mixlora.layers.{layer.layer_id_}.experts.{expert_idx}.{layer_name}.lora_B.weight"
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
                    lora_a_name = f"{name_prefix}.{layer.layer_id_}.self_attn.{layer_name}.lora_A.weight"
                    lora_b_name = f"{name_prefix}.{layer.layer_id_}.self_attn.{layer_name}.lora_B.weight"

                    if lora_a_name not in weight:
                        raise f"can not found the layer {lora_a_name} in model"
                    if lora_b_name not in weight:
                        raise f"can not found the layer {lora_b_name} in model"
                    lora_a = weight[lora_a_name]
                    lora_b = weight[lora_b_name]

                linear_layer_list[idx].init_lora_weight(
                    config, (lora_a, lora_b))


class OutputSequentialWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.wrapper_module_ = module

    def name(self) -> str:
        return type(self.wrapper_module_).__name__

    def forward(self, input: Tuple) -> Tuple:
        output = self.wrapper_module_.forward(input[0], input[2])
        return (output, ) + input[1:]


class LLMModel(torch.nn.Module):
    def __init__(self, model: LLMForCausalLM):
        super().__init__()
        args: LLMModelArgs = model.config_
        if args.vocab_size_ >= torch.finfo(args.dtype_).max:
            logging.warn(
                f"vocab_size >= max({args.dtype_}), consider load model with higher precision.")
        self.model_ = model
        self.config_ = args
        # configs
        self.name_or_path_ = args.name_or_path_
        self.vocab_size_ = args.vocab_size_
        self.device_ = args.device_
        self.dtype_ = args.dtype_

        self.output_ = OutputLayer()
        seq_module = model.sequential_module()
        seq_module.update({"output": OutputSequentialWrapper(self.output_)})
        seq_module.move_to_end("output")
        self.seq_module_ = torch.nn.Sequential(seq_module)
        # adapter configs
        self.adapter_configs_: Dict[str, LoraConfig] = {}

    # compute the model: output probs
    def forward(self, input_args: MultiLoraBatchData) -> List[LLMModelOutput]:
        assert input_args.batch_tokens_ is not None

        labels = input_args.batch_labels_

        # prepare inputs
        if isinstance(input_args.batch_tokens_, torch.Tensor):
            tokens = input_args.batch_tokens_.to(
                dtype=torch.int64, device=self.device_)
        else:
            tokens = torch.tensor(input_args.batch_tokens_,
                                  dtype=torch.int64, device=self.device_)

        # prepare mask
        if input_args.attention_masks_ is not None and 0 in input_args.attention_masks_:
            # 2d mask is passed through the layers
            if isinstance(input_args.attention_masks_, torch.Tensor):
                attn_mask = input_args.attention_masks_.to(
                    dtype=torch.int64, device=self.device_)
            else:
                attn_mask = torch.tensor(input_args.attention_masks_,
                                         dtype=torch.int64, device=self.device_)
        else:
            attn_mask = None

        if self.config_.attn_implementation_ != "flash_attn":
            causal_mask = self.model_.causal_mask(
                input_tokens=tokens,
                additional_mask=input_args.attention_masks_,
                diagonal=input_args.diagonal_pos_)
        else:
            causal_mask = attn_mask

        input_args.batch_labels_ = None
        input_args.batch_tokens_ = None
        input_args.attention_masks_ = None

        data = (tokens, causal_mask, input_args)

        for seq_layer in self.seq_module_:
            data = seq_layer.forward(data)

        # collecting router logits
        router_logits = [[]
                         for _ in range(len(input_args.lora_batch_data_config_))]
        for seq_layer in self.seq_module_:
            if not hasattr(seq_layer, "router_probs_"):
                continue
            for idx in range(len(input_args.lora_batch_data_config_)):
                if seq_layer.router_probs_[idx] is not None:
                    router_logits[idx].append(seq_layer.router_probs_[idx])

        # calculate loss
        output = data[0]
        assert isinstance(output, List)
        for idx, lora_config in enumerate(input_args.lora_batch_data_config_):
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
            if not input_args.output_router_logits_ or len(router_logits[idx]) == 0:
                continue
            # compute router loss when router logits is available
            loss_fn = router_loss_factory(
                self.adapter_configs_[output_data.adapter_name])
            if loss_fn is not None:
                output_data.aux_loss = loss_fn(router_logits[idx], attn_mask)

        return output

    def init_lora_layer_weight(self, config: LoraConfig, weight: Optional[Dict[str, torch.Tensor]]):
        self.adapter_configs_[config.adapter_name] = config
        # init output layer
        if config.task_name in task_dict and isinstance(task_dict[config.task_name], SequenceClassificationTask):
            output_layer = ClassificationOutputLayer(
                **task_dict[config.task_name].init_kwargs(),
                hidden_size=self.config_.dim_,
                pad_token_id=self.config_.pad_token_id_,
                device=self.device_,
                weight=weight)
        else:
            output_layer = CasualOutputLayer(
                vocab_size=self.config_.vocab_size_,
                weight=self.model_.lm_head_)

        self.output_.layers_[config.adapter_name] = output_layer
        if config.adapter_name == "default" and weight is None:
            return
        # init transformer layers
        for transformer_layer in self.model_.layers_:
            init_lora_layer_weight(
                transformer_layer, self.config_, config, weight)

    def from_pretrained(path: str,
                        device: str,
                        bits: int = None,
                        attn_impl: str = "eager",
                        use_sliding_window: bool = False,
                        load_dtype: torch.dtype = torch.bfloat16,
                        compute_dtype: torch.dtype = torch.bfloat16,
                        double_quant: bool = True,
                        quant_type: str = 'nf4'):
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
            llm_model = AutoModelForCausalLM.from_pretrained(
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
            llm_model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map=device,
                torch_dtype=load_dtype)

        llm_model.requires_grad_(False)

        model = from_pretrained(llm_model, attn_impl=attn_impl,
                                use_sliding_window=use_sliding_window, device=device)

        logging.info(f"Use {attn_impl} as attention implementation.")

        return LLMModel(model)

    def get_lora_weight_dict(self, lora_name: str) -> Dict[str, torch.Tensor]:
        # return the lora weight and target_module's name
        lora_weight_dict = self.output_.layers_[lora_name].state_dict()
        for idx, transformer_layer in enumerate(self.model_.layers_):
            if isinstance(self.adapter_configs_[lora_name], MixConfig):
                layer_prefix_name = f"mixlora.layers.{idx}.self_attn."
            else:
                layer_prefix_name = f"base_model.model.model.layers.{idx}.self_attn."

            lora_layer_list = []
            lora_layer_name_list = []
            for name, layer in transformer_layer.state_dict().items():
                lora_layer_list.append(layer)
                lora_layer_name_list.append(name)

            for idx, lora_layer in enumerate(lora_layer_list):
                if lora_name in lora_layer.loras_:
                    prefix_name = layer_prefix_name + lora_layer_name_list[idx]
                    lora_weight_dict[f"{prefix_name}.lora_A.weight"] = lora_layer.loras_[
                        lora_name].lora_a_.weight
                    lora_weight_dict[f"{prefix_name}.lora_B.weight"] = lora_layer.loras_[
                        lora_name].lora_b_.weight
                elif lora_name in transformer_layer.mlp_.moes_:
                    moe_layer_prefix_name = f"mixlora.layers.{transformer_layer.layer_id_}."
                    for expert_idx in range(transformer_layer.mlp_.moes_[lora_name].experts_):
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
                    ] = transformer_layer.mlp_.moes_[lora_name].gate_.weight

        return lora_weight_dict

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
