from mlora.pipeline.transport import RpcTransport
from mlora.pipeline.function import SendOperator, RecvOperator
from mlora.pipeline.messages import PipeMessageType
from mlora.pipeline.stream import CudaStream
from mlora.utils import setup_seed

import torch
import uuid
import argparse
import logging

from enum import Enum, auto
from typing import Dict

parser = argparse.ArgumentParser(description='Pipeline test case')
parser.add_argument('--device', type=str, default='cuda:0', required=True,
                    help='Specify which GPU to be used, default is cuda:0')
parser.add_argument('--rank', type=int, default=-1, required=True,
                    help="The device's rank number")
args = parser.parse_args()

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


class Pipe():
    world_size_: int = -1
    rank_: int = -1
    device_: torch.device = None
    role_: WorkerRole = None

    forward_stop_: bool = False
    backward_cache_: Dict[int, torch.Tensor] = {}
    stop_signal_: torch.tensor = None

    def is_stop_signal(self, data: torch.tensor) -> bool:
        if torch.numel(data) != 1:
            return False
        # is non zero will raise error if numel data != 1
        return not torch.is_nonzero(data)

    def __init__(self,
                 device: torch.device,
                 rank: int) -> None:

        self.world_size_ = torch.cuda.device_count()
        self.rank_ = rank
        self.device_ = device

        if rank == 0:
            self.role_ = WorkerRole.HEAD
        elif rank == self.world_size_ - 1:
            self.role_ = WorkerRole.TAIL
        else:
            self.role_ = WorkerRole.MID

        self.transport_ = RpcTransport(
            self.rank_, self.world_size_, self.device_)

        self.weight_ = torch.rand(
            (4096, 4096), dtype=torch.float32, device=self.device_, requires_grad=True)
        self.datas_ = [torch.rand(
            (4096, 4096), device=self.device_, dtype=torch.float32)] * G_TEST_TOTAL_CNT

        self.stop_signal_ = torch.tensor([0.], device=self.device_)
        assert self.is_stop_signal(self.stop_signal_)

        self.default_stream_ = CudaStream(
            torch.cuda.default_stream(self.device_))

    def run(self):
        while True:
            if self.role_ != WorkerRole.TAIL:
                self.process_backward()

            if self.role_ == WorkerRole.HEAD and not self.forward_stop_:
                self.process_input()

            if self.role_ != WorkerRole.HEAD and not self.forward_stop_:
                self.process_forward()

            if self.forward_stop_ and len(self.backward_cache_) == 0:
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
            self.forward_stop_ = True
            data = self.stop_signal_.clone().detach()
            logging.info("Forward done be signaled.")
        else:
            data = RecvOperator.apply(
                torch.tensor(1.0, requires_grad=True), self.transport_, message)
            data.grad_fn.pre_stage_fn = self.default_stream_.poll
            for _ in range(0, G_CALC_TOTAL_CNT):
                data = data @ self.weight_

        # mid worker need to send the result to next worker
        if self.role_ != WorkerRole.TAIL:
            self.default_stream_.poll()
            return self.send_next_worker(data)

        # tail worker need to calc the backward
        if not self.forward_stop_:
            logging.info(f"Calc the grad {data.sum()}.")
            data.sum().backward()

    def process_input(self):
        assert self.role_ == WorkerRole.HEAD
        assert not self.forward_stop_

        global G_TEST_CNT

        if G_TEST_CNT >= G_TEST_TOTAL_CNT:
            self.forward_stop_ = True
            data = self.stop_signal_.clone().detach()
            logging.info("Forward done be signaled.")
        else:
            logging.info(f"Train input data {G_TEST_CNT}.")
            data = self.datas_[G_TEST_CNT]
            for _ in range(0, G_CALC_TOTAL_CNT):
                data = data @ self.weight_

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
    rank = args.rank
    device = args.device
    setup_seed(42)
    pipe = Pipe(torch.device(device), rank)
    pipe.run()
