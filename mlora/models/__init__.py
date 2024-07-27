from .modeling_chatglm import GLMForCausalLM
from .modeling_gemma import GemmaForCausalLM
from .modeling_gemma2 import Gemma2ForCausalLM
from .modeling_llama import LlamaForCausalLM
from .modeling_mistral import MistralForCausalLM
from .modeling_mistral import MistralForCausalLM as Qwen2ForCausalLM
from .modeling_phi import PhiForCausalLM
from .modeling_phi3 import Phi3ForCausalLM

model_dict = {
    "llama": LlamaForCausalLM,
    "gemma": GemmaForCausalLM,
    "gemma2": Gemma2ForCausalLM,
    "mistral": MistralForCausalLM,
    "qwen2": Qwen2ForCausalLM,
    "phi": PhiForCausalLM,
    "phi3": Phi3ForCausalLM,
    "chatglm": GLMForCausalLM,
}


def from_pretrained(llm_model, **kwargs):
    if llm_model.config.model_type in model_dict:
        return model_dict[llm_model.config.model_type].from_pretrained(
            llm_model, **kwargs
        )
    else:
        raise RuntimeError(f"Model {llm_model.config.model_type} not supported.")


__all__ = [
    "LlamaForCausalLM",
    "GemmaForCausalLM",
    "MistralForCausalLM",
    "Qwen2ForCausalLM",
    "PhiForCausalLM",
    "Phi3ForCausalLM",
    "from_pretrained",
    "GLMForCausalLM",
]
