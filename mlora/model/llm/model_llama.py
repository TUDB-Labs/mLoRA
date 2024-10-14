import torch
from torch.nn.modules import Sequential
from transformers import AutoConfig, AutoModelForCausalLM
from typing import List, Optional, Tuple, Dict


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
    torch.Tensor,  # the mask matrices
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
        assert module_name in forward_func_dict, f"error module name {module_name}"

        return forward_func_dict[module_name]()


class LlamaModel(LLMModel):
    seq_module_: torch.nn.Sequential

    def __init__(self, args: LLMModelArgs):
        self.name_or_path_: str = args.name_or_path_
        self.norm_eps_ = args.norm_eps_
        self.device_ = args.device_
        self.n_heads_ = args.n_heads_
        self.dim_ = args.dim_
        self.vocab_size_ = args.vocab_size_
        self.pad_token_id_ = args.pad_token_id_
        self.eos_token_id_ = -1

    def forward(self, input: ModelData) -> torch.Tensor:
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

 
    @staticmethod
    def from_pretrained(
        path: str,
        device: str,
        precision: str,
        partial_model_to_device: Optional[List[int]] = None,
    ) -> LLMModel:
        def create_device_map() -> str | Dict[str, str]:
            if partial_model_to_device is None:
                return device
            else:
                config = AutoConfig.from_pretrained(path)
                weight_map = [
                    "model.embed_tokens",
                    *[f"model.layers.{layer_id}" for layer_id in range(0, config.num_hidden_layers)],
                    "model.norm",
                    "lm_head",
                ]
                return {map_item: "disk" for map_item in weight_map}

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
        additional_load_args["torch_dtype"] = load_type_dict.get(precision, torch.float32)
        llama_model = AutoModelForCausalLM.from_pretrained(path, **additional_load_args)

        llama_args = LLMModelArgs(llama_model.config)
        llama_args.pad_token_id_ = llama_args.pad_token_id_ or -1
        llama_args.device_ = device

        return LlamaModel.convert_model_from_huggingface(llama_model, llama_args)

    @staticmethod
    def convert_model_from_huggingface(llama_model: AutoModelForCausalLM, llama_args: LLMModelArgs):
        llama_model.requires_grad_(False)

        seq_model = OrderedDict()
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

    def load_adapter(self, adapter_model: AdapterModel):
        for module in self.seq_module_:
            if module.name() == "Decoder":
                module.wrapper_module_.load_adapter(adapter_model)

    def offload_adapter(self, adapter_name: str):
        for module in self.seq_module_:
            if module.name() == "Decoder":
                module.wrapper_module_.offload_adapter(adapter_name)

    def forward(self, input: ModelData) -> torch.Tensor:

        ret_val = OrderedDict()
        for module in self.seq_module_:
            if module.name() == "Decoder":
                ret_val.update(module.wrapper_module_.linears_info())
        return ret_val

    def sequential(self) -> Sequential:
        return self.seq_module_

    # New methods for applying LoRA rank and enabling specific layers
    def apply_lora(self, rank):
        """
        Apply the LoRA adapter with the specified rank to the model layers.
        """
        for layer in self.seq_module_:
            if isinstance(layer.wrapper_module_, Decoder):  
                layer.wrapper_module_.apply_lora(rank=rank)

    def enable_layers(self, enabled_layers):
        """
        Enable specific layers for LoRA adaptation.
        """
        if enabled_layers == 'last_2':
            for layer in self.seq_module_[-2:]:
                layer.wrapper_module_.enable_lora()
        elif enabled_layers == 'all':
            for layer in self.seq_module_:
                layer.wrapper_module_.enable_lora()
        elif enabled_layers == 'specific':
            specific_layers = [1, 3, 5]  
            for i, layer in enumerate(self.seq_module_):
                if i in specific_layers:
                    layer.wrapper_module_.enable_lora()
