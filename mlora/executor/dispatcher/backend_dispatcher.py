import logging
import threading
from typing import override

from mlora.config.dispatcher import DispatcherConfig
from mlora.config.task import TaskConfig

from .dispatcher import Dispatcher


class BackendDispatcher(Dispatcher):
    sem_: threading.Semaphore

    def __init__(self, config: DispatcherConfig) -> None:
        super().__init__(config)
        self.sem_ = threading.Semaphore(0)

    @override
    def add_task(self, config: TaskConfig, llm_name: str):
        super().add_task(config, llm_name)
        self.sem_.release()

    @override
    def is_done(self) -> bool:
        while len(self.running_) == 0 and len(self.ready_) == 0:
            # block until some task be add to the queue
            logging.info("Dispatcher no task, wait...")
            self.sem_.acquire()
        return False
