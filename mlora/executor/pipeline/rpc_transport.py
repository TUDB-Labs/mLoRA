import logging
import os
import queue
import uuid
from threading import Thread
from typing import Any, Dict, override

import torch
import torch.distributed.rpc

from .messages import PipeMessage, PipeMessageType
from .queue import DeviceSwapQueue
from .transport import Transport

# save by different message type
# recv/send queue will automatically change the tensors' device
RPCMessageRecvQueues: Dict[PipeMessageType, DeviceSwapQueue] = {}

RPCMessageSendQueues: Dict[PipeMessageType, DeviceSwapQueue] = {}

RPCCOMMMessageRecvQueues: Dict[PipeMessageType, queue.Queue] = {}

RPCCOMMMessageSendQueues: Dict[PipeMessageType, queue.Queue] = {}


def rpc_push_device_swap_queue(msg: PipeMessage) -> None:
    global RPCMessageRecvQueues

    assert (
        msg.msg_type_ in RPCMessageRecvQueues
    ), f"No this message type: {msg.msg_type_.value}"
    assert RPCMessageRecvQueues[msg.msg_type_] is not None

    logging.debug(f"RpcTransport async recv the message: {str(msg.msg_id_)[:8]}.")
    RPCMessageRecvQueues[msg.msg_type_].put(msg)


def rpc_push_comm_queue(msg: PipeMessage) -> None:
    global RPCCOMMMessageRecvQueues

    assert (
        msg.msg_type_ in RPCCOMMMessageRecvQueues
    ), f"No this comm message type: {msg.msg_type_.value}"
    assert RPCCOMMMessageRecvQueues[msg.msg_type_] is not None

    logging.debug(f"RpcTransport async recv the comm message: {str(msg.msg_id_)[:8]}.")
    RPCCOMMMessageRecvQueues[msg.msg_type_].put(msg)


# rpc transport thread
class RpcTransport(Transport):
    rank_: int
    world_size_: int
    worker_device_: torch.device

    stop_: bool
    activations_send_thread_: Thread
    gradients_send_thread_: Thread
    comm_send_thread_: Thread

    def __init__(self, rank: int, world_size: int, worker_device: torch.device) -> None:
        super().__init__(rank, world_size, worker_device)

        self.stop_: bool = False

        self.__init_device_swap_queue()
        self.__init_comm_queue()
        self.__init_background_thread()
        self.__init_rpc()

    def __init_rpc(self) -> None:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"

        assert self.rank_ > -1
        assert self.world_size_ > -1
        assert self.worker_device_ is not None

        # will be block when all world size's gpu join the group
        torch.distributed.rpc.init_rpc(
            f"worker-{self.rank_}", rank=self.rank_, world_size=self.world_size_
        )

        logging.info(f"Init rpc with rank {self.rank_} world_size: {self.world_size_}")

    def __init_device_swap_queue(self):
        cpu_device = torch.device("cpu")

        global RPCMessageSendQueues
        for key in [PipeMessageType.ACTIVATIONS, PipeMessageType.GRADIENTS]:
            RPCMessageSendQueues[key] = DeviceSwapQueue(
                self.worker_device_, cpu_device, queue_name=f"{key.value}_send"
            )
            RPCMessageSendQueues[key].start()

        global RPCMessageRecvQueues
        for key in [PipeMessageType.ACTIVATIONS, PipeMessageType.GRADIENTS]:
            RPCMessageRecvQueues[key] = DeviceSwapQueue(
                cpu_device, self.worker_device_, queue_name=f"{key.value}_recv"
            )
            RPCMessageRecvQueues[key].start()

    def __init_comm_queue(self):
        global RPCCOMMMessageSendQueues
        for key in [PipeMessageType.COMM]:
            RPCCOMMMessageSendQueues[key] = queue.Queue()

        global RPCCOMMMessageRecvQueues
        for key in [PipeMessageType.COMM]:
            RPCCOMMMessageRecvQueues[key] = queue.Queue()

    def __init_background_thread(self):
        self.gradients_send_thread_ = Thread(
            target=self.__send_loop, args=(PipeMessageType.GRADIENTS,)
        )
        self.activations_send_thread_ = Thread(
            target=self.__send_loop, args=(PipeMessageType.ACTIVATIONS,)
        )
        self.comm_send_thread_ = Thread(
            target=self.__comm_send_loop, args=(PipeMessageType.COMM,)
        )

        self.gradients_send_thread_.start()
        self.activations_send_thread_.start()
        self.comm_send_thread_.start()

    def __send_loop(self, msg_type: PipeMessageType):
        global RPCMessageSendQueues
        send_queue: DeviceSwapQueue = RPCMessageSendQueues[msg_type]
        assert send_queue is not None

        while not self.stop_ or not send_queue.empty():
            msg = send_queue.get_waitime()
            if msg is None:
                continue
            assert msg.tensor_data_ is not None
            assert msg.tensor_data_.device == torch.device("cpu")
            logging.debug(
                f"RpcTransport async send the message: {str(msg.msg_id_)[:8]} "
                f"to {msg.dst_}."
            )
            torch.distributed.rpc.rpc_async(
                msg.dst_, rpc_push_device_swap_queue, args=(msg,)
            )

    def __comm_send_loop(self, msg_type: PipeMessageType):
        global RPCCOMMMessageSendQueues
        send_queue: queue.Queue = RPCCOMMMessageSendQueues[msg_type]
        assert send_queue is not None

        while not self.stop_ or not send_queue.empty():
            try:
                msg = send_queue.get(block=True, timeout=10)
            except Exception:
                continue

            logging.debug(
                f"RpcTransport async send the message: {str(msg.msg_id_)[:8]}"
                f" to {msg.dst_}."
            )
            torch.distributed.rpc.rpc_async(msg.dst_, rpc_push_comm_queue, args=(msg,))

    def __stop_send_loop(self):
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
        self.comm_send_thread_.join()

    def __stop_rpc(self):
        torch.distributed.rpc.shutdown()

    def stop(self):
        self.__stop_send_loop()
        self.__stop_rpc()

    @override
    def recv_message(
        self, msg_type: PipeMessageType, block: bool = False
    ) -> PipeMessage | None:
        global RPCMessageRecvQueues

        assert msg_type in RPCMessageRecvQueues
        recv_queue: DeviceSwapQueue = RPCMessageRecvQueues[msg_type]

        if block:
            return recv_queue.get()
        else:
            return recv_queue.get_nowait()

    @override
    def send_message(self, msg: PipeMessage, sync: bool = False) -> None:
        assert not sync, "RPC transport do not suppose sync == true!"

        global RPCMessageSendQueues
        assert msg.msg_type_ in RPCMessageSendQueues
        send_queue: DeviceSwapQueue = RPCMessageSendQueues[msg.msg_type_]
        send_queue.put(msg)

    @override
    def recv_comm(self, msg_type: PipeMessageType, block: bool = False) -> PipeMessage:
        global RPCCOMMMessageRecvQueues

        assert msg_type in RPCCOMMMessageRecvQueues
        recv_queue: queue.Queue = RPCCOMMMessageRecvQueues[msg_type]

        if block:
            return recv_queue.get()
        else:
            return recv_queue.get_nowait()

    @override
    def send_comm(
        self, msg_type: PipeMessageType, data: Any, sync: bool = False
    ) -> None:
        pass
        assert not sync, "RPC transport do not suppose sync == true!"

        msg_id = uuid.uuid4().int

        msg = PipeMessage(
            src_=self.worker_name,
            dst_=self.next_worker_name,
            msg_type_=msg_type,
            msg_id_=msg_id,
            tensor_data_=None,
            model_data_=None,
            comm_data_=data,
        )

        global RPCCOMMMessageSendQueues
        assert msg.msg_type_ in RPCCOMMMessageSendQueues

        send_queue: queue.Queue = RPCCOMMMessageSendQueues[msg.msg_type_]
        send_queue.put(msg)
