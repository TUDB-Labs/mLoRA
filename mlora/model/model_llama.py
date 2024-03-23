from mlora.common import nvtx_wrapper
from mlora.config import LoraConfig
from mlora.model.modelargs import LLMModelArgs, MultiLoraBatchData
from mlora.model.model import LLMModel, repeat_kv, apply_rotary_emb, precompute_rope_angle, precompute_mask
from mlora.model.LoraLiner import Linear, Lora
from mlora.model.RMSNorm import RMSNorm
from mlora.model.Embedding import Embedding
from mlora.checkpoint.recompute import CheckpointRecomputeFunction

import logging
import torch
import torch.nn.functional as F
import xformers.ops
import xformers.ops.fmha.attn_bias

from transformers import AutoModelForCausalLM, AutoConfig
from typing import List, Dict, Tuple, Optional, Set
from collections import OrderedDict


class OutputLayer(torch.nn.Module):
    def __init__(self, args: LLMModelArgs, weight: torch.Tensor):
        super().__init__()
        self.lm_head_ = torch.nn.Linear(
            args.dim_, args.vocab_size_, bias=False, device=args.device_, dtype=args.dtype_)

        with torch.no_grad():
            if weight.device == torch.device('meta'):
                self.lm_head_.weight = weight
            else:
                self.lm_head_.weight.copy_(weight)
        self.lm_head_.requires_grad_(False)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.lm_head_(data).float()


class Transformer(torch.nn.Module):
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
        self.ffn_norm_: RMSNorm = None        # dim
        # rope angle cos and sin
        self.cos_, self.sin_ = precompute_rope_angle(
            args.dim_ // args.n_heads_, args.max_seq_len_,
            args.rope_theta_, args.device_)
        # other arg
        self.layer_id_ = layer_id
        self.norm_eps_ = args.norm_eps_
        self.n_heads_ = args.n_heads_
        self.n_kv_heads_ = args.n_kv_heads_
        self.n_rep_ = self.n_heads_ // self.n_kv_heads_
        self.head_dim_ = args.dim_ // args.n_heads_

    @property
    def linear_layer_name_to_module_dict(self) -> Dict[str, Linear]:
        # warnning: do not use the function when not init the linear layer
        ret = {"k_proj": self.wk_,
               "q_proj": self.wq_,
               "v_proj": self.wv_,
               "o_proj": self.wo_,
               "w1_proj": self.w1_,
               "w2_proj": self.w2_,
               "w3_proj": self.w3_}
        assert all([isinstance(layer, Linear) for _, layer in ret.items()])
        return ret

    def lora_layer_name(self,
                        name: str,
                        is_lora_a: bool = False,
                        is_lora_b: bool = False) -> str:
        assert is_lora_a ^ is_lora_b

        LORA_A_NAME_STR = "lora_A"
        LORA_B_NAME_STR = "lora_B"
        lora_layer_name_fmt = "base_model.model.model.layers.{}.self_attn.{}.{}.weight"

        return lora_layer_name_fmt.format(self.layer_id_, name, LORA_A_NAME_STR if is_lora_a else LORA_B_NAME_STR)

    def from_pretrained(self,
                        transformer_layer: torch.nn.Module,
                        norm_eps: float) -> None:
        linear_dict_name = {"wq_": transformer_layer.self_attn.q_proj,
                            "wk_": transformer_layer.self_attn.k_proj,
                            "wv_": transformer_layer.self_attn.v_proj,
                            "wo_": transformer_layer.self_attn.o_proj,
                            "w1_": transformer_layer.mlp.gate_proj,
                            "w2_": transformer_layer.mlp.down_proj,
                            "w3_": transformer_layer.mlp.up_proj}
        norm_dict_name = {"attention_norm_": transformer_layer.input_layernorm.weight,
                          "ffn_norm_": transformer_layer.post_attention_layernorm.weight}

        for var_dict_name, source in linear_dict_name.items():
            self.__dict__[var_dict_name] = Linear(source)

        for var_dict_name, source in norm_dict_name.items():
            self.__dict__[var_dict_name] = RMSNorm(source, norm_eps)

    def init_lora_layer_weight(self, lora_config: LoraConfig,
                               weight: Optional[Dict[str, torch.Tensor]]):
        # init the lora layer, if the weight state dict have already
        # exist this lora weight, use it, otherwise init it with zero
        name_module_dict: Dict[str,
                               Linear] = self.linear_layer_name_to_module_dict

        for name, module in name_module_dict.items():
            assert isinstance(module, Linear)

            if name in lora_config.target_ and lora_config.target_[name]:
                lora_weight = (None, None)
                lora_a_name = self.lora_layer_name(name, is_lora_a=True)
                lora_b_name = self.lora_layer_name(name, is_lora_b=True)

                if weight is not None and lora_a_name in weight:
                    assert lora_b_name in weight, f"can not found the layer {lora_b_name} in model."
                    lora_weight = (weight[lora_a_name], weight[lora_b_name])

                # init the lora layer
                module.init_lora_weight(lora_config, lora_weight)

    def forward(self,
                data: torch.Tensor,
                mask: torch.Tensor,
                input_args: MultiLoraBatchData):
        batch_size, max_seq_len, _ = data.shape

        attention_norm_data = self.attention_norm_.forward(data)

        xq = self.wq_.forward(attention_norm_data, input_args)
        xk = self.wk_.forward(attention_norm_data, input_args)
        xv = self.wv_.forward(attention_norm_data, input_args)

        # conver shape to multi head
        # the shape is batch_size * number_of_head * seq_len * dim_of_head
        xq = xq.view(batch_size, max_seq_len, self.n_heads_,
                     self.head_dim_).transpose(1, 2)
        xk = xk.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)
        xv = xv.view(batch_size, max_seq_len, self.n_kv_heads_,
                     self.head_dim_).transpose(1, 2)

        # apply rotary embedding
        assert xq.dtype == xk.dtype
        cos = self.cos_[:max_seq_len].to(xq.dtype)
        sin = self.sin_[:max_seq_len].to(xq.dtype)
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        # for llama2 need to repeat the heads
        # before dim: batch_size, n_kv_head, seq_len, head_dim
        # after dim: batch_size, n_head, seq_len, head_dim
        xk = repeat_kv(xk, self.n_rep_)
        xv = repeat_kv(xv, self.n_rep_)

        # must align with xformers memory efficient attention
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
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

    def get_lora_weight_dict(self, lora_name: str) -> Tuple[Dict[str, torch.Tensor], Set[str]]:
        lora_weight_dict = {}
        target_modules = set([])

        for name, module in self.linear_layer_name_to_module_dict.items():
            loras: Dict[str, Lora] = module.loras_
            if lora_name not in loras:
                continue
            if name not in target_modules:
                target_modules.add(name)
            lora: Lora = loras[lora_name]
            lora_weight_dict[self.lora_layer_name(name, is_lora_a=True)] = lora.lora_a_
            lora_weight_dict[self.lora_layer_name(name, is_lora_b=True)] = lora.lora_b_

        return lora_weight_dict, target_modules


LlamaSequentialModuleIO = Tuple[torch.Tensor,                         # the input batch tokens
                                torch.Tensor,                         # the mask matrics
                                MultiLoraBatchData,                   # batch data config
                                bool                                  # whether to use checkpoint
                                ]
LEN_LLAMA_SEQUENTIAL_MODULE_IO = 4

LlamaCompatibleModelTypes = ["mistral", "qwen2"]


class LlamaSequentialWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.wrapper_module_ = module

    def name(self) -> str:
        return type(self.wrapper_module_).__name__

    def forward(self, input: LlamaSequentialModuleIO) -> LlamaSequentialModuleIO:
        assert isinstance(input, Tuple)
        assert len(input) == LEN_LLAMA_SEQUENTIAL_MODULE_IO
        assert isinstance(input[0], torch.Tensor)
        assert isinstance(input[1], torch.Tensor)
        assert isinstance(input[2], MultiLoraBatchData)
        assert isinstance(input[3], bool)

        # auto catch the input argument
        def embedding_forward():
            output = self.wrapper_module_.forward(input[0])
            if input[-1]:
                output = output.requires_grad_(True)
            return (output, ) + input[1:]

        def transformer_forward():
            if input[-1]:
                output = CheckpointRecomputeFunction.apply(
                    self.wrapper_module_.forward, *input[:-1])
            else:
                output = self.wrapper_module_.forward(*input[:-1])
            return (output, ) + input[1:]

        def rmsnorm_forward():
            output = self.wrapper_module_.forward(input[0])
            return (output, ) + input[1:]

        def output_layer_forward():
            output = self.wrapper_module_.forward(input[0])
            return (output, ) + input[1:]

        forward_func_dict = {
            "Embedding": nvtx_wrapper(embedding_forward, "embedding"),
            "Transformer": nvtx_wrapper(transformer_forward, "transformer"),
            "RMSNorm": nvtx_wrapper(rmsnorm_forward, "rmsnorm"),
            "OutputLayer": nvtx_wrapper(output_layer_forward, "output"),
        }

        module_name = self.name()
        assert module_name in forward_func_dict, f"error module name {module_name}"

        return forward_func_dict[module_name]()


class LlamaModel(LLMModel):
    def __init__(self, args: LLMModelArgs):
        self.name_or_path_ = args.name_or_path_
        # weight
        self.token_embedding_: Embedding = None

        self.layers_: List[Transformer] = []
        for layer_id in range(args.n_layers_):
            self.layers_.append(Transformer(layer_id, args))

        self.norm_: RMSNorm = None         # dim
        self.output_: OutputLayer = None   # vocab size * dim

        # sequential model
        self.seq_module_: torch.nn.Sequential = None

        self.norm_eps_ = args.norm_eps_

        self.device_ = args.device_
        self.n_heads_ = args.n_heads_
        self.vocab_size_ = args.vocab_size_
        self.pad_token_id_ = args.pad_token_id_
        self.dim_ = args.dim_

        # need to set
        self.eos_token_id_ = -1

    def forward(self, input: MultiLoraBatchData) -> torch.Tensor:
        # train model or inference model: output is probs
        tokens = torch.tensor(input.batch_tokens_,
                              dtype=torch.int64,
                              device=self.device_)

        mask = precompute_mask(tokens, self.n_heads_,
                               self.device_, input.additional_mask_)

        if input.inference_model_:
            data = (tokens, mask, input, False)
        else:
            data = (tokens, mask, input, True)

        for seq_layer in self.seq_module_:
            data = seq_layer.forward(data)

        return data[0]

    def init_lora_weight(self,
                         lora_config: LoraConfig,
                         weight: Optional[Dict[str, torch.Tensor]]):
        for transformer_layer in self.layers_:
            transformer_layer.init_lora_layer_weight(lora_config, weight)

    @staticmethod
    def from_pretrained(path: str,
                        device: str,
                        bits: int = None,
                        # below argument just for qlora (4-bit quant)
                        qlora_4bit_fp16: bool = True,
                        qlora_4bit_bf16: bool = False,
                        qlora_4bit_double_quant: bool = True,
                        qlora_4_bit_quant_type: str = "nf4",
                        partial_model_to_device: List[int] = None) -> LLMModel:
        assert qlora_4bit_fp16 ^ qlora_4bit_bf16
        assert (bits is None) or (bits in [4, 8])
        assert qlora_4_bit_quant_type in ["nf4", "fp4"]

        # create the device map for parallelism
        def create_device_map():
            if partial_model_to_device is None:
                device_map = device
            else:
                config = AutoConfig.from_pretrained(path)
                # Be careful, this is hard coded.
                weight_map = ["model.embed_tokens",
                              *[f"model.layers.{layer_id}" for layer_id in range(0, config.num_hidden_layers)],
                              "model.norm",
                              "lm_head"]
                device_map = {map_item: "disk" for map_item in weight_map}
                for partial_weight in partial_model_to_device:
                    device_map[weight_map[partial_weight]] = device
            return device_map

        # the argument for the LlamaForCausalLM load the pretrained large model
        additional_load_args = {
            "device_map": create_device_map(),
            "torch_dtype": torch.float32,
        }

        if bits is not None:
            logging.info('Loading model with quantization, bits = %i' % bits)
            from transformers import BitsAndBytesConfig
            qlora_4bit_compute_dtype = torch.float32
            # if set the compute type, then change it, otherwise hold the default
            qlora_4bit_compute_dtype = torch.float16 if qlora_4bit_fp16 else qlora_4bit_compute_dtype
            qlora_4bit_compute_dtype = torch.bfloat16 if qlora_4bit_bf16 else qlora_4bit_compute_dtype

            torch_dtype = torch.float32
            torch_dtype = torch.bfloat16 if qlora_4bit_bf16 else torch_dtype
            additional_load_args["torch_dtype"] = torch_dtype
            additional_load_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True if bits == 4 else False,
                load_in_8bit=True if bits == 8 else False,
                llm_int8_enable_fp32_cpu_offload=True,
                # only for qlora 4bit
                bnb_4bit_compute_dtype=qlora_4bit_compute_dtype,
                bnb_4bit_use_double_quant=qlora_4bit_double_quant,
                bnb_4bit_quant_type=qlora_4_bit_quant_type,
            )

        llama_model = AutoModelForCausalLM.from_pretrained(
            path, **additional_load_args)

        if llama_model.config.model_type != "llama":
            if llama_model.config.model_type in LlamaCompatibleModelTypes:
                logging.info(
                    f"loading {llama_model.config.model_type} model with llama compatible mode.")
            else:
                logging.warning(
                    f"unsupported model type {llama_model.config.model_type}, loading with llama compatible mode.")

        llama_args = LLMModelArgs(llama_model.config)
        if llama_args.pad_token_id_ is None:
            llama_args.pad_token_id_ = -1
        llama_args.device_ = device
        llama_args.dtype_ = llama_model.dtype

        # load model from pretrained large model
        model = LlamaModel.convert_model_from_huggingface(
            llama_model, llama_args)

        # convert to sequential module for use
        model.seq_module_ = model.sequential_module()

        return model

    @staticmethod
    def convert_model_from_huggingface(llama_model: AutoModelForCausalLM,
                                       llama_args: LLMModelArgs):
        model = LlamaModel(llama_args)

        llama_model.requires_grad_(False)

        # plm - pretrained large model
        def get_tensor_from_plm(name: str) -> torch.Tensor:
            origin_weight_map = {"embedding": llama_model.model.embed_tokens.weight,
                                 "norm": llama_model.model.norm.weight,
                                 "output": llama_model.lm_head.weight}
            assert name in origin_weight_map
            origin_weight = origin_weight_map[name]
            assert isinstance(origin_weight, torch.Tensor)
            return origin_weight

        model.token_embedding_ = Embedding(
            get_tensor_from_plm("embedding"), llama_args.pad_token_id_)

        for idx, target_layer in enumerate(llama_model.model.layers):
            assert isinstance(model.layers_[idx], Transformer)
            target_transformer: Transformer = model.layers_[idx]
            target_transformer.from_pretrained(target_layer, model.norm_eps_)

        model.norm_ = RMSNorm(
            get_tensor_from_plm("norm"), model.norm_eps_)

        model.output_ = OutputLayer(llama_args, get_tensor_from_plm("output"))

        return model

    def get_train_paramas(self) -> Dict[str, List[torch.Tensor]]:
        # warnning: this will return all the lora's parameters

        # the lora layer inside the linear layer
        all_linear_layer_name = ["wq_", "wk_",
                                 "wv_", "wo_", "w1_", "w2_", "w3_"]

        def get_all_linear_layer(layer: Transformer):
            assert isinstance(layer, Transformer), f"error type {type(layer)}"
            # transformer in disk do not return the train paramas
            if layer.w1_.device_ == torch.device("meta"):
                logging.debug(
                    f"Layer-{layer.layer_id_} do not be load in the worker, skip.")
                return []

            # all linear layer from this transformer layer
            all_linear_layer: List[Linear] = [layer.__dict__[linear_layer_name]
                                              for linear_layer_name in all_linear_layer_name]
            return all_linear_layer

        def get_all_loras_layer(layer: Linear):
            assert isinstance(layer, Linear), f"error type {type(layer)}"
            # all lora adapter from this linear layer
            return layer.loras_

        all_linear_layer = [linear_layer
                            for transformer_layer in self.layers_
                            for linear_layer in get_all_linear_layer(transformer_layer)]

        all_loras_layer: List[Dict[str, Lora]] = [get_all_loras_layer(
            linear_layer) for linear_layer in all_linear_layer]

        train_paramas = {}
        for loras in all_loras_layer:
            for adatper_name, lora in loras.items():
                if adatper_name not in train_paramas:
                    train_paramas[adatper_name] = []
                train_paramas[adatper_name].extend(
                    (lora.lora_a_, lora.lora_b_))

        return train_paramas

    def get_lora_weight_dict(self, lora_name: str) -> Tuple[Dict[str, torch.Tensor], Set[str]]:
        # return the lora weight dict and target lora module's name
        #   for example, lora_weight_dict = {"self_atten.q_proj.lora_A.weight", tensor}
        #                target_modules   = ["q_proj", "k_proj"]
        lora_weight_dict = {}
        target_modules = set([])

        # each transformer layer
        for transformer_layer in self.layers_:
            lora_weight, target_module = transformer_layer.get_lora_weight_dict(lora_name)
            lora_weight_dict.update(lora_weight)
            target_modules.update(target_module)

        return lora_weight_dict, target_modules

    def sequential_module(self) -> torch.nn.Sequential:
        seq_module = OrderedDict()

        # must ensure the follow order
        seq_module.update(
            {"embedding": LlamaSequentialWrapper(self.token_embedding_)})

        for index, layer in enumerate(self.layers_):
            layer_name = f"layer{index}"
            seq_module.update({layer_name: LlamaSequentialWrapper(layer)})

        seq_module.update({"norm": LlamaSequentialWrapper(self.norm_)})

        seq_module.update({"output": LlamaSequentialWrapper(self.output_)})

        return torch.nn.Sequential(seq_module)
