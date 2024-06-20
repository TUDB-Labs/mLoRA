from mlora.config import MLoRAConfig, TaskConfig
from mlora.model.llm import LLMModel
from mlora.model.tokenizer import Tokenizer
from mlora.model.args import MLoRAData

import torch
import logging
from typing import Dict, Callable

from .dispatcher import Dispatcher, DISPATCHER_CLASS
from .task import Task


class Executor:
    model_: LLMModel = None
    tokenizer_: Tokenizer = None

    dispatcher_: Dispatcher = None

    def __init__(self,
                 model: LLMModel,
                 tokenizer: Tokenizer,
                 config: MLoRAConfig) -> None:
        self.model_ = model
        self.tokenizer_ = tokenizer

        dispatcher_name = config.dispatcher_.name_
        assert dispatcher_name in DISPATCHER_CLASS
        self.dispatcher_ = DISPATCHER_CLASS[dispatcher_name](
            config.dispatcher_)

        hook_func = {
            "init": self.__task_init_hook,
            "running": self.__task_to_running_hook,
            "ready": self.__task_to_ready_hook,
            "done": self.__task_to_done_hook,
        }

        for hook, cb in hook_func.items():
            self.dispatcher_.register_hook(hook, cb)

    def register_hook(self, name: str, cb: Callable):
        self.dispatcher_.register_hook(name, cb)

    def __task_init_hook(self, task: Task):
        logging.info(
            f"Init {task.task_type()} : {task.task_name()} task with adapters: {task.adapter_name()}")
        # init the task's dataset
        # init the task's adapter weight
        task.prepare(self.model_.linears_info(), self.tokenizer_)

    def __task_to_running_hook(self, task: Task):
        logging.info(
            f"Base model load adapters: {task.adapter_name()}")
        # move the task's adapter weight to the gpu
        # move the task's optimizer weight to the gpu
        # attach the adapter to the model
        # NOTE: must ensure the weight be loaded in the device befor attach to the model
        task.switch_device(self.model_.device_)
        for adapter_model in task.adapter_model():
            self.model_.load_adapter(adapter_model)

    def __task_to_ready_hook(self, task: Task):
        logging.info(
            f"Base model offload adapters: {task.adapter_name()}")
        # offload the adapter
        # move the task's adapter weight to the cpu
        for adapter_name in task.adapter_name():
            self.model_.offload_adapter(adapter_name)
        task.switch_device("cpu")

    def __task_to_done_hook(self, task: Task):
        logging.info(
            f"Finish and base model offload adapter - {task.adapter_name()}")
        # offload the adapter
        # move the task's adapter weight to the cpu
        for adapter_name in task.adapter_name():
            self.model_.offload_adapter(adapter_name)
        task.switch_device("cpu")
        task.done()

    def dispatcher_info(self) -> Dict[str, str]:
        return self.dispatcher_.info()

    def add_task(self, config: TaskConfig):
        self.dispatcher_.add_task(config, self.model_.name_or_path_)

    def execute(self) -> None:
        while not self.dispatcher_.is_done():
            data: MLoRAData = self.dispatcher_.data()

            output = self.model_.forward(data.model_data())
            labels = torch.tensor(data.batch_tokens_, dtype=torch.long)

            total_loss = None

            for config in data.data_config_:
                loss = config.loss_fn_(
                    output, labels, torch.tensor(data.batch_mask_))
                if loss is None:
                    continue
                total_loss = loss if total_loss is None else total_loss + loss

            total_loss.backward()

            self.dispatcher_.step()
