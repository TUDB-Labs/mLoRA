from mlora.config import MLoRAConfig, TaskConfig
from mlora.model.llm import LLMModel
from mlora.model.tokenizer import Tokenizer

import torch
import logging

from .dispatcher import Dispatcher
from .task import Task


class Trainer:
    model_: LLMModel = None
    tokenizer_: Tokenizer = None

    dispatcher_: Dispatcher = None

    def __init__(self,
                 model: LLMModel,
                 tokenizer: Tokenizer,
                 config: MLoRAConfig) -> None:
        self.model_ = model
        self.tokenizer_ = tokenizer

        self.dispatcher_ = Dispatcher(config.dispather_)

        hook_func = {
            "init": self.__task_init_hook,
            "running": self.__task_to_running_hook,
            "ready":  self.__task_to_ready_hook,
            "done": self.__task_to_done_hook,
        }

        for hook, cb in hook_func.items():
            self.dispatcher_.register_hook(hook, cb)

    def __task_init_hook(self, task: Task):
        logging.info(f"Init task - {task.config_.adapter_.name_}")
        # init the task's dataset
        # init the task's adapter weight
        # init the task's optimizer state
        # init the task's lr scheduler state
        task.pre_init(self.model_.linears_info(), self.tokenizer_)

    def __task_to_running_hook(self, task: Task):
        logging.info(
            f"Base model load adapter - {task.config_.adapter_.name_}")
        # move the task's adapter weight to the gpu
        # move the task's optimizer weight to the gpu
        # attach the adapter to the model
        # NOTE: must ensure the weight be loaded in the device befor attach to the model
        task.switch_device(self.model_.device_)
        self.model_.load_adapter(task.context_)

    def __task_to_ready_hook(self, task: Task):
        logging.info(
            f"Base model offload adapter - {task.config_.adapter_.name_}")
        # offload the adapter
        # move the task's adapter weight to the cpu
        self.model_.offload_adapter(task.config_.adapter_.name_)
        task.switch_device("cpu")

    def __task_to_done_hook(self, task: Task):
        logging.info(
            f"Finish adapter - {task.config_.adapter_.name_}")
        # offload the adapter
        # move the task's adapter weight to the cpu
        self.model_.offload_adapter(task.config_.adapter_.name_)
        task.switch_device("cpu")
        # to save the model
        task.save()

    def add_task(self, config: TaskConfig):
        self.dispatcher_.add_task(config, self.model_.name_or_path_)

    def train(self) -> None:
        while not self.dispatcher_.is_done():
            data, loss_fns = self.dispatcher_.get_train_data()

            output = self.model_.forward(data)

            total_loss = None
            labels = torch.tensor(data.batch_tokens_, dtype=torch.long)
            for idx, config in enumerate(data.data_config_):
                s_idx = config.batch_start_idx_
                e_idx = config.batch_end_idx_
                vocab_size = output.shape[-1]
                loss_input = output[s_idx:e_idx][...,
                                                 :-1, :].contiguous().view(-1, vocab_size)
                loss_target = labels[s_idx:e_idx][...,
                                                  1:].contiguous().view(-1).to(loss_input.device)
                loss = loss_fns[idx](loss_input, loss_target)
                total_loss = loss if total_loss is None else total_loss + loss
                logging.info(f"Task - {config.adapter_name_} loss: {loss}")

            total_loss.backward()

            self.dispatcher_.step()
