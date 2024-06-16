from .task import Task
from .train_task import TrainTask
from .dpo_task import DPOTask


TASK_CLASS = {
    "train": TrainTask,
    "dpo": DPOTask
}

__all__ = [
    "Task",
    "TASK_CLASS",
    "TrainTask",
    "DPOTask",
]
