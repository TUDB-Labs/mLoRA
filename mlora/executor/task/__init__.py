from .task import Task
from .train_task import TrainTask
from .dpo_task import DPOTask
from .cpo_task import CPOTask


TASK_CLASS = {
    "train": TrainTask,
    "dpo": DPOTask,
    "cpo": CPOTask
}

__all__ = [
    "Task",
    "TASK_CLASS",
    "TrainTask",
    "DPOTask",
    "CPOTask"
]
