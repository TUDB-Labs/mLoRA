from .context import TaskContext
from .lora import TrainLoRAContext, InferenceLoRAContext
from .loraplus import TrainLoRAPlusContext

TRAINCONTEXT_CLASS = {
    "lora": TrainLoRAContext,
    "loraplus": TrainLoRAPlusContext
}

INFERENCECONTEXT_CLASS = {
    "lora": InferenceLoRAContext,
    "loraplus": InferenceLoRAContext
}

__all__ = [
    "TRAINCONTEXT_CLASS",
    "INFERENCECONTEXT_CLASS",
    "TaskContext",
    "TrainLoRAContext",
    "InferenceLoRAContext",
    "TrainLoRAPlusContext",
]
