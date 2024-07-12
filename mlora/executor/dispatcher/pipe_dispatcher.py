from typing import List, Set, override

from mlora.config.dispatcher import DispatcherConfig
from mlora.executor.task import Task
from mlora.model.args import Masks, MLoRAData, MLoRADataConfig, Tokens

from .backend_dispatcher import BackendDispatcher


class PipeDispatcher(BackendDispatcher):
    lock_set_: Set[str]

    def __init__(self, config: DispatcherConfig) -> None:
        super().__init__(config)
        self.lock_set_ = set()

    @override
    def _dispatch_task_in(self):
        # ready task to terminate
        terminate_task = [task for task in self.ready_ if task.is_terminate()]
        self.ready_ = [task for task in self.ready_ if not task.is_terminate()]

        for task in terminate_task:
            self.terminate_event_.notify(task)

        # pipeline only have one running task
        while len(self.running_) <= self.concurrency_num_ and len(self.ready_) > 0:
            task = self.ready_.pop(0)
            self.running_.append(task)
            self.running_event_.notify(task)

    def find_the_task(self, task_name: str) -> Task:
        # the worker do not really dispather the task
        # so we just find it in the read
        for task in self.ready_:
            if task.task_name() != task_name:
                continue
            return task
        raise Exception(f"No this task {task.task_name()}")

    # if not the head worker, we need to manully dispatch the task
    def dispatch_task_to_run(self, task_name: str):
        task = self.find_the_task(task_name)
        self.running_event_.notify(task)

    def dispatch_task_to_ready(self, task_name: str):
        task = self.find_the_task(task_name)
        self.ready_event_.notify(task)

    def dispatch_task_to_done(self, task_name: str):
        task = self.find_the_task(task_name)
        self.done_event_.notify(task)

    def dispatch_task_to_terminal(self, task_name: str):
        task = self.find_the_task(task_name)
        self.terminate_event_.notify(task)

    def dispatch_task_to_step(self, task_name: str):
        task = self.find_the_task(task_name)
        task.step()
        self.step_event_.notify(task)

    def lock_task(self, name: str):
        self.lock_set_.add(name)

    def unlock_task(self, name: str):
        if name not in self.lock_set_:
            return
        self.lock_set_.remove(name)

    def is_lock(self, name: str):
        return name in self.lock_set_

    @override
    def data(self) -> MLoRAData | None:
        self._dispatch_task_in()

        batch_tokens: List[Tokens] = []
        batch_masks: List[Masks] = []
        data_configs: List[MLoRADataConfig] = []

        can_run_task = list(
            filter(lambda task: not self.is_lock(task.task_name()), self.running_)
        )

        if len(can_run_task) == 0:
            return None

        # get all train data
        start_idx: int = 0
        # pipe dispatcher just run one task
        task = can_run_task[0]

        data, data_config = task.data(start_idx)

        # for unlock the task
        for item in data_config:
            item.task_name_ = task.task_name()

        data_configs.extend(data_config)
        batch_tokens.extend(data)
        start_idx = start_idx + len(data)
        self.lock_task(task.task_name())

        # post process this batch data
        batch_tokens, batch_masks = self._align_batch_tokens(batch_tokens, data_configs)

        return MLoRAData(
            batch_tokens=batch_tokens, batch_mask=batch_masks, data_config=data_configs
        )

    def task_step(self, task_name: str):
        # in head worker the task must in running
        for task in self.running_:
            if task.task_name() != task_name:
                continue
            task.step()
            self.step_event_.notify(task)

        self._dispatch_task_out()
