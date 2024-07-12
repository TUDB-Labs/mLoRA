from .backends import get_backend
from .common import (
    AdapterConfig,
    LLMBatchConfig,
    LLMForCausalLM,
    LLMModelArgs,
    LLMModelInput,
    LLMModelOutput,
    LoraConfig,
    MixConfig,
    lora_config_factory,
)
from .dispatcher import Dispatcher, TrainTask
from .evaluator import EvaluateConfig, evaluate
from .generator import GenerateConfig, generate
from .model import LLMModel
from .prompter import Prompter
from .tokenizer import Tokenizer
from .trainer import TrainConfig, train
from .utils import is_package_available, setup_logging

assert is_package_available("torch", "2.3.0"), "m-LoRA requires torch>=2.3.0"
assert is_package_available(
    "transformers", "4.42.0"
), "m-LoRA requires transformers>=4.42.0"


setup_logging()

__all__ = [
    "LLMModelArgs",
    "LLMModelOutput",
    "LLMForCausalLM",
    "LLMBatchConfig",
    "LLMModelInput",
    "AdapterConfig",
    "LoraConfig",
    "MixConfig",
    "lora_config_factory",
    "TrainTask",
    "Dispatcher",
    "EvaluateConfig",
    "evaluate",
    "GenerateConfig",
    "generate",
    "TrainConfig",
    "train",
    "LLMModel",
    "Prompter",
    "Tokenizer",
    "setup_logging",
    "get_backend",
]
