from aspen.modelargs import LLMModelArgs, MultiLoraBatchData
from aspen.model import LLMModel, RMSNorm
from aspen.LoraLiner import Linear

import torch

from transformers import AutoModel
from typing import List, Dict


class Transformer:
    def __init__(self, layer_id: int, args: LLMModelArgs):
        # feed forward - MLP
        self.dense_h_to_4h: Linear = None  # 4dim * dim
        self.dense_4h_to_h: Linear = None  # dim * 4dim
        # norm
        self.input_layer_norm_: RMSNorm = None  # dim
        self.post_layer_norm_: RMSNorm = None   # dim
        # other args
        self.layer_id_ = layer_id


class ChatGLMModel(LLMModel):
    def __init__(self, args: LLMModelArgs):
        # weight
        self.token_embedding_: torch.Tensor = None

        self.layers_: List[Transformer] = []
        for layer_id in range(args.n_layers_):
            self.layers_.append(Transformer(layer_id, args))

        self.norm_: RMSNorm = None
        self.output_: torch.Tensor = None  # vocab size * dim

        self.device_ = args.device

    def forward(self, input: MultiLoraBatchData):
        # tokens = torch.tensor(input.batch_tokens_,
        # dtype=torch.int64).to(self.dev)
        if input.inference_model_:
            raise NotImplementedError
        else:
            pass

    def from_pretrained(path: str,
                        device: str,
                        bits: int = None,
                        fp16: bool = True,
                        bf16: bool = True,
                        double_quant: bool = True,
                        quant_type: str = 'nf4',
                        log_fn=None):
        if bits in [4, 8]:
            if log_fn is not None:
                log_fn('Loading model with quantization, bits = %i' % bits)
            from transformers import BitsAndBytesConfig
            compute_dtype = (torch.float16 if fp16 else (
                torch.bfloat16 if bf16 else torch.float32))
            chatglm_model = AutoModel.from_pretrained(
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
                torch_dtype=(torch.float32 if fp16 else (
                    torch.bfloat16 if bf16 else torch.float32)),
                trust_remote_code=True)
        else:
            chatglm_model = AutoModel.from_pretrained(
                path,
                device_map=device,
                torch_dtype=torch.float32,
                trust_remote_code=True)

        print(chatglm_model)

        chatglm_args = LLMModelArgs()
        chatglm_args.device = device

        model = ChatGLMModel(chatglm_args)

        raise NotImplementedError

        return model

    def get_train_paramas(self, config: Dict[str, str]) -> Dict[str, List[torch.Tensor]]:
        pass
