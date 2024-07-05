from typing import Dict, Type

from .context import TaskContext
from .inference import InferenceTaskContext
from .lora import InferenceLoRAContext, TrainLoRAContext
from .loraplus import TrainLoRAPlusContext
from .train import TrainTaskContext
from .vera import InferenceVeRAContext, TrainVeRAContext

TRAINCONTEXT_CLASS: Dict[str, Type[TrainTaskContext]] = {
    "lora": TrainLoRAContext,
    "loraplus": TrainLoRAPlusContext,
    "vera": TrainVeRAContext,
}

INFERENCECONTEXT_CLASS: Dict[str, Type[InferenceTaskContext]] = {
    "lora": InferenceLoRAContext,
    "loraplus": InferenceLoRAContext,
    "vera": InferenceVeRAContext,
}


__all__ = [
    "TRAINCONTEXT_CLASS",
    "INFERENCECONTEXT_CLASS",
    "TaskContext",
    "TrainTaskContext",
    "TrainLoRAContext",
    "InferenceLoRAContext",
    "TrainLoRAPlusContext",
]
