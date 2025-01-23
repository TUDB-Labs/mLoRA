import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, override

import torch
from torch.nn.modules import Sequential
from transformers import AutoConfig, AutoModelForCausalLM

from mlora.model.args import LinearInfo, LLMModelArgs, Masks, ModelData
from mlora.model.checkpoint import CheckpointRecomputeFunction
from mlora.model.modules import AdapterModel, Decoder, Embedding, OutputLayer, RMSNorm
from mlora.profiler import nvtx_wrapper, set_backward_tracepoint
from mlora.utils import is_package_available

if is_package_available("bitsandbytes"):
    from transformers import BitsAndBytesConfig
else:
    from mlora.utils import BitsAndBytesConfig

from .model_llm import LLMModel


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
def precompute_mask(
    input_tokens: torch.Tensor,
    n_heads: int,
    device: str,
    additional_mask: List[Masks] | None = None,
    diagonal: int = 1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if input_tokens.dim() == 2:
        batch_size, seq_len = input_tokens.shape
    elif input_tokens.dim() == 3:
        batch_size, seq_len, _ = input_tokens.shape
    else:
        raise Exception("input dim is not correct {input_tokens.dim}")

    TORCH_MIN_VALUE = torch.finfo(dtype).min
    mask = torch.full(
        (batch_size, n_heads, seq_len, seq_len),
        TORCH_MIN_VALUE,
        device=device,
        dtype=dtype,
    )
    mask = torch.triu(mask, diagonal=diagonal)

    if additional_mask is not None:
        masks_metric = torch.tensor(additional_mask, dtype=torch.bool, device=device)
        masks_metric = masks_metric.view(batch_size, 1, 1, seq_len)
        masks_metric = masks_metric.expand(-1, n_heads, seq_len, -1)
        mask = torch.masked_fill(mask, masks_metric, TORCH_MIN_VALUE)

    mask.requires_grad_(False)

    return mask.to(device=device, dtype=dtype)


LlamaSequentialModuleIO = Tuple[
    torch.Tensor,  # the input batch tokens
    torch.Tensor,  # the mask matrics
    ModelData,  # batch data config
    bool,  # whether to use checkpoint
]
LEN_LLAMA_SEQUENTIAL_MODULE_IO = 4

LlamaCompatibleModelTypes = ["mistral", "qwen2", "llama"]


class LlamaSequentialWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.wrapper_module_ = module

    def name(self) -> str:
        return type(self.wrapper_module_).__name__

    def forward(self, input: LlamaSequentialModuleIO) -> LlamaSequentialModuleIO:
        assert len(input) == LEN_LLAMA_SEQUENTIAL_MODULE_IO
        assert isinstance(input[0], torch.Tensor)
        assert isinstance(input[1], torch.Tensor)
        assert isinstance(input[2], ModelData)
        assert isinstance(input[3], bool)

        # auto catch the input argument
        @nvtx_wrapper("f_embedding")
        def embedding_forward():
            output = self.wrapper_module_.forward(input[0])
            if input[-1]:
                output = output.requires_grad_(True)
            return (output,) + input[1:]

        def decoder_forward():
            if input[-1]:
                output = CheckpointRecomputeFunction(
                    self.wrapper_module_.forward, *input[:-1]
                )
                set_backward_tracepoint(output.grad_fn, "b_checkpoint")
            else:
                output = self.wrapper_module_.forward(*input[:-1])
            return (output,) + input[1:]

        @nvtx_wrapper("f_rmsnorm")
        def rmsnorm_forward():
            output = self.wrapper_module_.forward(input[0])
            set_backward_tracepoint(output.grad_fn, "b_rmsnorm")
            return (output,) + input[1:]

        @nvtx_wrapper("f_output")
        def output_layer_forward():
            output = self.wrapper_module_.forward(input[0])
            set_backward_tracepoint(output.grad_fn, "b_output")
            return (output,) + input[1:]

        forward_func_dict = {
            "Embedding": embedding_forward,
            "Decoder": decoder_forward,
            "RMSNorm": rmsnorm_forward,
            "OutputLayer": output_layer_forward,
        }

        module_name = self.name()
        assert (
            module_name in forward_func_dict
        ), f"error module name {
            module_name}"

        return forward_func_dict[module_name]()


class LlamaModel(LLMModel):
    seq_module_: torch.nn.Sequential

    def __init__(self, args: LLMModelArgs):
        self.name_or_path_: str = args.name_or_path_
        # sequential model

        self.norm_eps_ = args.norm_eps_

        self.device_ = args.device_
        self.n_heads_ = args.n_heads_
        self.dim_ = args.dim_
        self.vocab_size_ = args.vocab_size_

        # need to set
        self.pad_token_id_ = args.pad_token_id_
        self.eos_token_id_ = -1

    @override
    def forward(self, input: ModelData) -> torch.Tensor:
        # train model or inference model: output is probs
        tokens = torch.tensor(
            input.batch_tokens_, dtype=torch.int64, device=self.device_
        )

        mask = precompute_mask(tokens, self.n_heads_, self.device_, input.batch_mask_)

        if input.enable_checkpoint_:
            data = (tokens, mask, input, True)
        else:
            data = (tokens, mask, input, False)

        for seq_layer in self.seq_module_:
            data = seq_layer.forward(data)

        return data[0]

    @override
    @staticmethod
    def from_pretrained(
        path: str,
        device: str,
        precision: str,
        partial_model_to_device: Optional[List[int]] = None,
    ) -> LLMModel:
        # create the device map for parallelism
        def create_device_map() -> str | Dict[str, str]:
            device_map: str | Dict[str, str]
            if partial_model_to_device is None:
                device_map = device
            else:
                config = AutoConfig.from_pretrained(path)
                # Be careful, this is hard coded.
                weight_map = [
                    "model.embed_tokens",
                    *[
                        f"model.layers.{layer_id}"
                        for layer_id in range(0, config.num_hidden_layers)
                    ],
                    "model.norm",
                    "lm_head",
                ]
                device_map = {map_item: "disk" for map_item in weight_map}
                for partial_weight in partial_model_to_device:
                    device_map[weight_map[partial_weight]] = device
            return device_map

        # the argument for the LlamaForCausalLM load the pretrained large model
        load_type_dict = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }

        additional_load_args = {
            "device_map": create_device_map(),
            "torch_dtype": torch.float32,
        }

        logging.info(f"Loading model with precision - {precision}")

        if precision in load_type_dict:
            additional_load_args["torch_dtype"] = load_type_dict[precision]
        else:
            load_4bit = precision in ["nf4", "fp4"]
            load_8bit = precision == "int8"

            additional_load_args["torch_dtype"] = torch.float32
            additional_load_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=load_4bit,
                load_in_8bit=load_8bit,
                # int8 only for GPU, fp32 for cpu
                llm_int8_enable_fp32_cpu_offload=True,
                # do not hold the fp16 part
                # when forward and backward need to convert int8 to fp16
                llm_int8_has_fp16_weight=False,
                # only for qlora 4bit
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type=precision,
            )

        llama_model = AutoModelForCausalLM.from_pretrained(path, **additional_load_args)

        if llama_model.config.model_type not in LlamaCompatibleModelTypes:
            assert f"unsupported model type {
                llama_model.config.model_type}, loading with llama compatible mode."

        logging.info(
            f"loading llama compatible model - {llama_model.config.model_type}"
        )

        llama_args = LLMModelArgs(llama_model.config)
        if llama_args.pad_token_id_ is None:
            llama_args.pad_token_id_ = -1
        llama_args.device_ = device
        llama_args.dtype_ = llama_model.dtype

        # load model from pretrained large model
        model = LlamaModel.convert_model_from_huggingface(llama_model, llama_args)

        return model

    @staticmethod
    def convert_model_from_huggingface(
        llama_model: AutoModelForCausalLM, llama_args: LLMModelArgs
    ):
        llama_model.requires_grad_(False)

        seq_model: OrderedDict[str, torch.nn.Module] = OrderedDict()

        seq_model.update(
            {
                "embedding": LlamaSequentialWrapper(
                    Embedding(
                        llama_model.model.embed_tokens.weight, llama_args.pad_token_id_
                    )
                )
            }
        )

        for idx, target_layer in enumerate(llama_model.model.layers):
            decoder = Decoder(idx, llama_args)
            decoder.from_pretrained(target_layer, llama_args.norm_eps_)
            seq_model.update({f"layer{idx}": LlamaSequentialWrapper(decoder)})

        seq_model.update(
            {
                "norm": LlamaSequentialWrapper(
                    RMSNorm(llama_model.model.norm.weight, llama_args.norm_eps_)
                )
            }
        )
        seq_model.update(
            {
                "output": LlamaSequentialWrapper(
                    OutputLayer(llama_model.lm_head.weight, llama_args)
                )
            }
        )

        model = LlamaModel(llama_args)
        model.seq_module_ = torch.nn.Sequential(seq_model)

        return model

    @override
    def load_adapter(self, adapter_model: AdapterModel):
        # module is LlamaSequentialWrapper
        for module in self.seq_module_:
            if module.name() != "Decoder":
                continue
            module.wrapper_module_.load_adapter(adapter_model)

    @override
    def offload_adapter(self, adapter_name: str):
        # now only transformers block have adapter
        for module in self.seq_module_:
            if module.name() != "Decoder":
                continue
            module.wrapper_module_.offload_adapter(adapter_name)

    @override
    def linears_info(self) -> OrderedDict[str, LinearInfo]:
        ret_val = OrderedDict()
        for module in self.seq_module_:
            if module.name() != "Decoder":
                continue
            ret_val.update(module.wrapper_module_.linears_info())
        return ret_val

    @override
    def sequential(self) -> Sequential:
        return self.seq_module_
