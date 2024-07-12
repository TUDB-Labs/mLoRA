from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.distributed.rpc

from .messages import PipeMessage, PipeMessageType


class Transport(ABC):
    rank_: int
    world_size_: int
    worker_device_: torch.device

    @property
    def next_worker_name(self) -> str:
        return f"worker-{(self.rank_ + 1) % self.world_size_}"

    @property
    def prev_worker_name(self) -> str:
        return f"worker-{(self.rank_ - 1) % self.world_size_}"

    @property
    def worker_name(self) -> str:
        return f"worker-{self.rank_}"

    @abstractmethod
    def recv_message(
        self, msg_type: PipeMessageType, block: bool = False
    ) -> PipeMessage | None:
        pass

    @abstractmethod
    def send_message(self, msg: PipeMessage, sync: bool = False) -> None:
        pass

    @abstractmethod
    def recv_comm(self, msg_type: PipeMessageType, block: bool = False) -> PipeMessage:
        pass

    @abstractmethod
    def send_comm(
        self, msg_type: PipeMessageType, data: Any, sync: bool = False
    ) -> None:
        pass

    def __init__(self, rank: int, world_size: int, worker_device: torch.device) -> None:
        self.rank_ = rank
        self.world_size_ = world_size
        self.worker_device_ = worker_device
