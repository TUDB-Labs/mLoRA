from .context import TaskContext
from .lora import LoRATaskContext

TASKCONTEXT_CLASS = {
    "lora": LoRATaskContext
}

__all__ = [
    "TASKCONTEXT_CLASS",
    "TaskContext",
    "LoRATaskContext"
]
