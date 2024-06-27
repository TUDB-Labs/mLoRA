from .cpo_task import CPOTask
from .dpo_task import DPOTask
from .task import Task
from .train_task import TrainTask

TASK_CLASS = {"train": TrainTask, "dpo": DPOTask, "cpo": CPOTask}

__all__ = ["Task", "TASK_CLASS", "TrainTask", "DPOTask", "CPOTask"]
