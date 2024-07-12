import logging
from queue import Queue
from threading import Thread
from typing import Optional

import torch

from .messages import PipeMessage
from .stream import CudaStream


class DeviceSwapQueue:
    copy_stream_: CudaStream

    def __init__(
        self,
        source_device: torch.device,
        target_device: torch.device,
        target_size: int = 0,
        queue_name: str = "default",
    ) -> None:
        source_device_is_cpu: bool = (
            True if source_device == torch.device("cpu") else False
        )
        target_device_is_cpu: bool = (
            True if target_device == torch.device("cpu") else False
        )

        assert source_device_is_cpu ^ target_device_is_cpu

        if source_device_is_cpu:
            self.copy_stream_: CudaStream = CudaStream(torch.cuda.Stream(target_device))
        else:
            self.copy_stream_: CudaStream = CudaStream(torch.cuda.Stream(source_device))

        self.target_device_: torch.device = target_device
        self.source_device_: torch.device = source_device
        # TODO: change the size by the size of avaliable gpu memory
        self.src_queue_: Queue = Queue()
        self.dst_queue_: Queue = Queue(target_size)

        self.queue_name_: str = queue_name

        self.stop_: bool = False

    def swap_thread_loop(self):
        try:
            msg: PipeMessage = self.src_queue_.get(block=True, timeout=0.001)
        except Exception:
            return
        logging.debug(
            f"{self.queue_name_} swap the message - {str(msg.msg_id_)[:8]} start."
        )

        # must ensure the msg.tensor_data_ sync done
        with torch.cuda.stream(self.copy_stream_.stream_):
            # do not use the pined_memory maybe can speedup
            # need more test
            assert msg.tensor_data_ is not None
            copy_tensor = (
                torch.zeros_like(msg.tensor_data_, device=self.target_device_)
                .copy_(msg.tensor_data_, non_blocking=True)
                .detach()
            )
            msg.tensor_data_ = copy_tensor
            # msg.tensor_data_ = msg.tensor_data_.to(
            #     self.target_device_, non_blocking=True).detach()

        self.copy_stream_.poll()

        logging.debug(
            f"{self.queue_name_} swap the message - {str(msg.msg_id_)[:8]} device end."
        )
        self.dst_queue_.put(msg, block=True)

    def swap_thread(self):
        logging.info(f"DeviceSwapQueue - {self.queue_name_} start.")
        while not self.stop_ or not self.src_queue_.empty():
            self.swap_thread_loop()
        logging.info(f"DeviceSwapQueue - {self.queue_name_} stop.")

    def start(self):
        self.swap_thread_ = Thread(target=self.swap_thread)
        self.swap_thread_.start()

    def stop(self):
        self.stop_ = True
        self.swap_thread_.join()

    def get(self) -> PipeMessage:
        return self.dst_queue_.get(block=True)

    def get_waitime(self, timeout: int = 10) -> Optional[PipeMessage]:
        try:
            return self.dst_queue_.get(block=True, timeout=timeout)
        except Exception:
            return None

    def get_nowait(self) -> Optional[PipeMessage]:
        try:
            return self.dst_queue_.get_nowait()
        except Exception:
            return None

    def put(self, msg: PipeMessage):
        self.src_queue_.put(msg)

    def empty(self) -> bool:
        return self.src_queue_.empty() and self.dst_queue_.empty()
