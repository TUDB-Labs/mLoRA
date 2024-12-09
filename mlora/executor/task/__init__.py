import logging
from typing import MutableMapping, Type

from .cit_task import CITTask
from .cpo_task import CPOTask
from .dpo_task import DPOTask
from .ppo_task import PPOTask
from .task import Task
from .train_task import TrainTask

TASK_CLASS: MutableMapping[str, Type[Task]] = {
    "train": TrainTask,
    "dpo": DPOTask,
    "cpo": CPOTask,
    "cit": CITTask,
    "ppo": PPOTask,
}


def register_task_class(type_name: str, task: Type[Task]):
    global TASK_CLASS

    if type_name in TASK_CLASS:
        logging.info(f"Task type {type_name} already exist skip register it.")
        return

    TASK_CLASS[type_name] = task


__all__ = [
    "Task",
    "TASK_CLASS",
    "TrainTask",
    "DPOTask",
    "CPOTask",
    "CITTask",
    "PPOTask",
    "register_task_class",
]
