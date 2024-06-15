from mlora.config.dispatcher import DispatcherConfig
from mlora.config.task import TaskConfig
from mlora.trainer.task import Task
from mlora.model.args import MLoRABatchData, Tokens, Masks, MLoRADataConfig

import math
import torch
from typing import List, Callable, Tuple


class DispatcherEvent():
    callback_list_: List[Callable] = None

    def __init__(self):
        self.callback_list_ = []

    def register(self, func: Callable):
        self.callback_list_.append(func)

    def notify(self, task: Task) -> None:
        for func in self.callback_list_:
            func(task)


class Dispatcher:
    name_: str = ""

    ready_: List[Task] = []
    running_: List[Task] = []
    done_: List[Task] = []

    init_event_: DispatcherEvent = DispatcherEvent()
    running_event_: DispatcherEvent = DispatcherEvent()
    ready_event_: DispatcherEvent = DispatcherEvent()
    done_event_: DispatcherEvent = DispatcherEvent()

    concurrency_num_: int = 2

    def __init__(self, config: DispatcherConfig) -> None:
        self.name_ = config.name_
        self.concurrency_num_ = config.concurrency_num_

    def register_hook(self, name: str, cb: Callable) -> None:
        event_map = {
            "init": self.init_event_,
            "running": self.running_event_,
            "ready": self.ready_event_,
            "done": self.done_event_
        }

        assert name in event_map

        event_map[name].register(cb)

    def add_task(self, config: TaskConfig, llm_name: str):
        task = Task(config, llm_name)
        self.ready_.append(task)
        self.init_event_.notify(task)

    def is_done(self) -> bool:
        return len(self.running_) == 0 and len(self.ready_) == 0

    def __dispatch_task_in(self):
        # ready task to running task
        assert len(self.running_) <= self.concurrency_num_
        if len(self.running_) == self.concurrency_num_:
            return

        while len(self.running_) < self.concurrency_num_ and len(self.ready_) > 0:
            task = self.ready_.pop(0)
            self.running_.append(task)
            self.running_event_.notify(task)

    def __dispatch_task_out(self):
        # running task to ready task or done task
        done_task = [task for task in self.running_ if task.is_done()]
        self.running_ = [task for task in self.running_ if not task.is_done()]
        self.done_.extend(done_task)
        for task in done_task:
            self.done_event_.notify(task)

    def __align_batch_tokens(self, batch_tokens: List[Tokens],
                             configs: List[MLoRADataConfig]) -> Tuple[List[Tokens], List[Masks]]:
        max_seq_len = max(map(lambda x: len(x), batch_tokens))
        max_seq_len = math.ceil(max_seq_len / 8) * 8

        batch_masks: List[Masks] = []

        for idx, task in enumerate(self.running_):
            data_config = configs[idx]
            assert data_config.adapter_name_ == task.context_.name_
            s_idx = data_config.batch_start_idx_
            e_idx = data_config.batch_end_idx_

            batch_tokens[s_idx:e_idx], masks = task.expand_batch_tokens(
                batch_tokens[s_idx:e_idx], max_seq_len)
            batch_masks.extend(masks)

        return batch_tokens, batch_masks

    def get_train_data(self) -> Tuple[MLoRABatchData, List[torch.nn.Module]]:
        self.__dispatch_task_in()

        batch_tokens: List[Tokens] = []
        batch_masks: List[Masks] = []
        data_configs: List[MLoRADataConfig] = []
        loss_fns: List[torch.nn.Module] = []

        # get all train data
        start_idx: int = 0
        for task in self.running_:
            data = task.get_train_data()
            end_idx = start_idx + len(data)
            data_configs.append(MLoRADataConfig(
                adapter_name_=task.context_.name_, adapter_type_=task.context_.type_,
                batch_start_idx_=start_idx, batch_end_idx_=end_idx))
            batch_tokens.extend(data)
            loss_fns.append(task.get_loss_fn())
            start_idx = end_idx

        # post process this batch data
        batch_tokens, batch_masks = self.__align_batch_tokens(
            batch_tokens, data_configs)

        return MLoRABatchData(batch_tokens_=batch_tokens, batch_mask_=batch_masks,
                              data_config_=data_configs, inference_model_=False), loss_fns

    def step(self):
        for _, task in enumerate(self.running_):
            task.step()

        self.__dispatch_task_out()
