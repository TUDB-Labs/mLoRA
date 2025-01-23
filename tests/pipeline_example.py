from mlora.executor.pipeline.rpc_transport import RpcTransport
from mlora.executor.pipeline.function import SendOperator, RecvOperator
from mlora.executor.pipeline.messages import PipeMessageType
from mlora.executor.pipeline.stream import CudaStream
from mlora.utils import setup_seed

import os
import torch
import uuid
import logging

from enum import Enum, auto
from typing import Dict, List


logging.basicConfig(format="[%(asctime)s] [%(threadName)s] m-LoRA: %(message)s",
                    level="INFO",
                    handlers=[logging.StreamHandler()],
                    force=True)

G_CALC_TOTAL_CNT = 10
G_TEST_TOTAL_CNT = 10
G_TEST_CNT = 0


class WorkerRole(Enum):
    HEAD = auto()
    MID = auto()
    TAIL = auto()


class TestModel(torch.nn.Module):
    def __init__(self, device: torch.device):
        super(TestModel, self).__init__()
        self.device_ = device
        self.weight_ = torch.rand(
            (4096, 4096), dtype=torch.float32, device=self.device_, requires_grad=True)

    def forward(self, data: torch.Tensor):
        for _ in range(0, G_CALC_TOTAL_CNT):
            data = data @ self.weight_
            # too big will make the grad to be inf
            data /= 1000
        return data


class Pipe():
    world_size_: int = -1
    rank_: int = -1
    device_: torch.device = None
    role_: WorkerRole = None

    forward_stop_: bool = False
    input_stop_: bool = False
    backward_cache_: Dict[int, torch.Tensor] = {}

    forward_cnt_: int = 0
    stop_signal_: torch.Tensor = None

    def is_stop_signal(self, data: torch.tensor) -> bool:
        return data.dtype == torch.long and torch.numel(data) == 1

    def __init__(self, rank: int, world_size: int, device: torch.device = None) -> None:
        self.world_size_ = world_size
        self.rank_ = rank
        self.device_ = device if device else torch.device(f"cuda:{self.rank_}")

        if rank == 0:
            self.role_ = WorkerRole.HEAD
        elif rank == self.world_size_ - 1:
            self.role_ = WorkerRole.TAIL
        else:
            self.role_ = WorkerRole.MID

        self.transport_ = RpcTransport(
            self.rank_, self.world_size_, self.device_)

        self.model_ = TestModel(self.device_)
        self.datas_ = [torch.rand(
            (4096, 4096), device=self.device_, dtype=torch.float32)] * G_TEST_TOTAL_CNT

        self.forward_cnt_ = 0
        self.forward_stop_ = False
        self.input_stop_ = False

        self.default_stream_ = CudaStream(
            torch.cuda.default_stream(self.device_))

        self.test_grads_: List[torch.Tensor] = []

    def run(self):
        if self.role_ == WorkerRole.HEAD:
            self.forward_stop_ = True
        if self.role_ != WorkerRole.HEAD:
            self.input_stop_ = True

        while True:
            if self.role_ != WorkerRole.TAIL:
                self.process_backward()

            if not self.input_stop_:
                self.process_input()

            if not self.forward_stop_:
                self.process_forward()

            if len(self.backward_cache_) == 0 and self.forward_stop_ and self.input_stop_:
                # no froward and backward request
                break

        logging.info("Pipe done and to stop.")
        # clear the pipeline resource
        self.stop()

    def stop(self):
        transport = self.transport_
        if isinstance(transport, RpcTransport):
            transport.stop()
        logging.info("Transport stop.")

    def process_backward(self):
        assert self.role_ != WorkerRole.TAIL

        message = self.transport_.recv_message(
            PipeMessageType.GRADIENTS, block=False)
        if message is None:
            return
        logging.info(
            f"Recv the gradients - {str(message.msg_id_)[:8]} from {message.src_}")

        msg_id = message.msg_id_

        assert msg_id in self.backward_cache_
        phony: torch.Tensor = self.backward_cache_[msg_id]
        phony.grad_fn.grad_from_next_worker = message.tensor_data_
        phony.backward()
        self.test_grads_.append(self.model_.weight_.grad.sum())

        del self.backward_cache_[msg_id]

    def process_forward(self):
        assert self.role_ != WorkerRole.HEAD
        assert not self.forward_stop_

        # recv the tensors from prev-worker
        message = self.transport_.recv_message(
            PipeMessageType.ACTIVATIONS, block=False)
        if message is None:
            return
        logging.info(
            f"Recv the activations - {str(message.msg_id_)[:8]} from {message.src_}")

        # use RecvOperator get the real data
        #   the operator also auto send the backward grad to prev worker
        if self.is_stop_signal(message.tensor_data_):
            self.stop_signal_ = message.tensor_data_
            data = message.tensor_data_
            logging.info("Forward done be signaled.")
        else:
            data = RecvOperator.apply(
                torch.tensor(1.0, requires_grad=True), self.transport_, message)
            data.grad_fn.pre_stage_fn = self.default_stream_.poll
            self.forward_cnt_ += 1
            data = self.model_(data)

        if self.stop_signal_ is not None and self.stop_signal_.item() == self.forward_cnt_:
            self.forward_stop_ = True

        # mid worker need to send the result to next worker
        if self.role_ != WorkerRole.TAIL:
            self.default_stream_.poll()
            return self.send_next_worker(data)

        # tail worker need to calc the backward
        if not self.forward_stop_:
            logging.info(f"Calc the grad {data.sum()}.")
            data.sum().backward()
            self.test_grads_.append(self.model_.weight_.grad.sum())

    def process_input(self):
        assert self.role_ == WorkerRole.HEAD
        assert not self.input_stop_

        global G_TEST_CNT

        if G_TEST_CNT >= G_TEST_TOTAL_CNT:
            self.input_stop_ = True
            data = torch.tensor(
                [self.forward_cnt_], dtype=torch.long, device="cpu", requires_grad=False)
            assert self.is_stop_signal(data)
            logging.info("Forward done be signaled.")
        else:
            logging.info(f"Train input data {G_TEST_CNT}.")
            self.forward_cnt_ += 1
            data = self.datas_[G_TEST_CNT]
            data = self.model_(data)

        G_TEST_CNT += 1

        self.default_stream_.poll()
        self.send_next_worker(data)

    def send_next_worker(self, tensor_data: torch.Tensor) -> None:
        assert isinstance(tensor_data, torch.Tensor)

        msg_id = uuid.uuid4().int
        assert msg_id not in self.backward_cache_

        if self.is_stop_signal(tensor_data):
            msg_id = -1

        phony: torch.Tensor = SendOperator.apply(torch.tensor(
            1.0, requires_grad=True), tensor_data, self.transport_, msg_id, None)

        if self.is_stop_signal(tensor_data):
            return

        self.backward_cache_[msg_id] = phony


if __name__ == "__main__":
    assert "RANK" in os.environ
    rank = int(os.environ["RANK"])
    setup_seed(42)
    pipe = Pipe(rank, torch.cuda.device_count())
    pipe.run()
