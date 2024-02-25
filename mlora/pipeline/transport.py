from mlora.pipeline.messages import PipeMessageType, PipeMessage
from mlora.pipeline.queue import DeviceSwapQueue

import os
import logging
import torch
import torch.distributed.rpc

from typing import Dict
from abc import ABC, abstractmethod
from threading import Thread

# save by different message type
# recv/send queue will automatically change the tensors' device
RPCMessageRecvQueues: Dict[PipeMessageType, DeviceSwapQueue] = {
    PipeMessageType.ACTIVATIONS: None,
    PipeMessageType.GRADIENTS: None
}

RPCMessageSendQueues: Dict[PipeMessageType, DeviceSwapQueue] = {
    PipeMessageType.ACTIVATIONS: None,
    PipeMessageType.GRADIENTS: None
}


def rpc_push_queue(msg: PipeMessage) -> None:
    global RPCMessageRecvQueues

    assert msg.msg_type_ in RPCMessageRecvQueues, f"No this message type: {msg.msg_type_.value}"
    assert RPCMessageRecvQueues[msg.msg_type_] is not None

    logging.debug(
        f"RpcTransport async recv the message: {str(msg.msg_id_)[:8]}.")
    RPCMessageRecvQueues[msg.msg_type_].put(msg)


class Transport(ABC):
    rank_: int
    device_: torch.device

    @property
    def next_worker_name(self) -> str:
        return f"worker-{self.rank_ + 1}"

    @property
    def prev_worker_name(self) -> str:
        return f"worker-{self.rank_ - 1}"

    @property
    def worker_name(self) -> str:
        return f"worker-{self.rank_}"

    @abstractmethod
    def recv_message(self, msg_type: PipeMessageType, block: bool = False) -> PipeMessage:
        pass

    @abstractmethod
    def send_message(self, msg: PipeMessage, sync: bool = False) -> None:
        pass


# rpc transport thread
class RpcTransport(Transport):
    rank_: int = -1
    world_size_: int = -1
    worker_device_: torch.device = None

    stop_: bool = False
    activations_send_thread_: Thread = None
    gradients_send_thread_: Thread = None

    def __init__(self, rank: int, world_size: int, worker_device: torch.device) -> None:
        self.rank_ = rank
        self.world_size_ = world_size
        self.worker_device_ = worker_device

        self.stop_: bool = False

        self.init_rpc()
        self.init_device_swap_queue()
        self.init_background_thread()

    def init_rpc(self) -> None:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"

        assert self.rank_ > -1
        assert self.world_size_ > -1
        assert self.worker_device_ is not None

        # will be block when all world size's gpu join the group
        torch.distributed.rpc.init_rpc(
            f"worker-{self.rank_}", rank=self.rank_, world_size=self.world_size_)

        logging.info(
            f"Init rpc with rank {self.rank_} world_size: {self.world_size_}")

    def init_device_swap_queue(self):
        cpu_device = torch.device("cpu")

        global RPCMessageSendQueues
        for key in RPCMessageSendQueues:
            RPCMessageSendQueues[key] = DeviceSwapQueue(
                self.worker_device_, cpu_device, queue_name=f"{key.value}_send")
            RPCMessageSendQueues[key].start()

        global RPCMessageRecvQueues
        for key in RPCMessageRecvQueues:
            RPCMessageRecvQueues[key] = DeviceSwapQueue(
                cpu_device, self.worker_device_, queue_name=f"{key.value}_recv")
            RPCMessageRecvQueues[key].start()

    def init_background_thread(self):
        self.gradients_send_thread_ = Thread(
            target=self.send_loop, args=(PipeMessageType.GRADIENTS,))
        self.activations_send_thread_ = Thread(
            target=self.send_loop, args=(PipeMessageType.ACTIVATIONS,))

        self.gradients_send_thread_.start()
        self.activations_send_thread_.start()

    def send_loop(self, msg_type: PipeMessageType):
        global RPCMessageSendQueues
        send_queue: DeviceSwapQueue = RPCMessageSendQueues[msg_type]
        assert send_queue is not None

        while not self.stop_ or not send_queue.empty():
            msg = send_queue.get_waitime()
            if msg is None:
                continue
            assert msg.tensor_data_.device == torch.device("cpu")
            logging.debug(
                f"RpcTransport async send the message: {str(msg.msg_id_)[:8]} to {msg.dst_}.")
            torch.distributed.rpc.rpc_async(
                msg.dst_, rpc_push_queue, args=(msg,))

    def stop_send_loop(self):
        global RPCMessageRecvQueues
        global RPCMessageSendQueues

        # first should stop the recv queue
        for key in RPCMessageRecvQueues:
            RPCMessageRecvQueues[key].stop()

        # then stop the send queue
        for key in RPCMessageSendQueues:
            RPCMessageSendQueues[key].stop()

        self.stop_ = True
        self.activations_send_thread_.join()
        self.gradients_send_thread_.join()

    def stop_rpc(self):
        torch.distributed.rpc.shutdown()

    def stop(self):
        self.stop_send_loop()
        self.stop_rpc()

    def recv_message(self, msg_type: PipeMessageType, block: bool = True) -> PipeMessage:
        global RPCMessageRecvQueues

        assert msg_type in RPCMessageRecvQueues
        recv_queue: DeviceSwapQueue = RPCMessageRecvQueues[msg_type]

        if block:
            return recv_queue.get()
        else:
            return recv_queue.get_nowait()

    def send_message(self, msg: PipeMessage, sync: bool = False) -> None:
        assert not sync, "RPC transport do not suppose sync == true!"

        global RPCMessageSendQueues
        assert msg.msg_type_ in RPCMessageSendQueues
        send_queue: DeviceSwapQueue = RPCMessageSendQueues[msg.msg_type_]
        send_queue.put(msg)
