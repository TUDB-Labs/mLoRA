from mlora.pipeline.transport import RpcTransport
from mlora.pipeline.stream import CudaStream
from mlora.pipeline.messages import PipeMessageType
from mlora.pipeline.function import RecvOperator, SendOperator
from mlora.model.model import LLMModel, precompute_mask
from mlora.model.modelargs import MultiLoraBatchData
from mlora.dispatcher import Dispatcher
from mlora.utils import create_optimizer_from, get_accumulation_steps

import torch
import uuid
import logging

from enum import Enum, auto
from typing import Dict, List
from collections.abc import Iterable


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
    input_stop_: bool = False
    forward_cnt_: int = 0
    backward_cache_: Dict[int, torch.Tensor] = {}
    stop_signal_: torch.tensor = None

    model_partition_: torch.nn.Sequential = torch.nn.Sequential()
    data_dispatcher_: Iterable[MultiLoraBatchData] = None
    n_heads_: int = -1

    config_: Dict[str, any] = {}

    loss_fn_: torch.nn.Module = torch.nn.CrossEntropyLoss()
    optimizer_: Dict[str, torch.optim.Optimizer] = {}
    accumulation_step_: Dict[str, int] = {}

    def is_stop_signal(self, data: torch.tensor) -> bool:
        return data.dtype == torch.long and torch.numel(data) == 1

    def __init__(self,
                 model: LLMModel,
                 config: Dict[str, any],
                 dispatcher: Dispatcher,
                 device: torch.device,
                 rank: int,
                 balance: List[int]) -> None:
        self.world_size_ = torch.cuda.device_count()
        assert self.world_size_ == len(balance)

        self.rank_ = rank
        self.device_ = device
        self.balance_ = balance

        if rank == 0:
            self.role_ = WorkerRole.HEAD
        elif rank == self.world_size_ - 1:
            self.role_ = WorkerRole.TAIL
        else:
            self.role_ = WorkerRole.MID

        self.transport_ = RpcTransport(
            self.rank_, self.world_size_, self.device_)

        self.default_stream_ = CudaStream(
            torch.cuda.default_stream(self.device_))

        self.config_ = config
        self.data_dispatcher_ = dispatcher.train_data()

        # need the config value, so must in the last stage to init
        self.init_partition(model)

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

    def process_input(self):
        assert self.role_ == WorkerRole.HEAD
        assert not self.input_stop_

        try:
            train_input = next(self.data_dispatcher_)
            tokens = torch.tensor(train_input.batch_tokens_,
                                  dtype=torch.int64,
                                  device=self.device_)
            data = self.forward(tokens, train_input)
            self.forward_cnt_ += 1
        except StopIteration:
            # stop
            self.input_stop_ = True
            train_input = None
            data = torch.tensor(
                [self.forward_cnt_], dtype=torch.long, device="cpu", requires_grad=False)
            assert self.is_stop_signal(data)
            logging.info("Forward done be signaled.")

        self.default_stream_.poll()
        self.send_next_worker(data, train_input)

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
            self.stop_signal_ = message.tensor_data_
            data = message.tensor_data_
            logging.info("Forward done be signaled.")
        else:
            data = RecvOperator.apply(
                torch.tensor(1.0, requires_grad=True), self.transport_, message)
            data.grad_fn.pre_stage_fn = self.default_stream_.poll
            self.forward_cnt_ += 1
            data = self.forward(data, message.batch_data_)

        if self.stop_signal_ is not None and self.stop_signal_.item() == self.forward_cnt_:
            self.forward_stop_ = True

        # mid worker need to send the result to next worker
        if self.role_ != WorkerRole.TAIL:
            self.default_stream_.poll()
            return self.send_next_worker(data, message.batch_data_)

        # tail worker need to calc the backward
        if not self.forward_stop_ and not self.is_stop_signal(message.tensor_data_):
            labels = torch.tensor(
                message.batch_data_.batch_tokens_,
                dtype=torch.long,
                device=self.device_)

    def send_next_worker(self,
                         tensor_data: torch.Tensor,
                         batch_data: MultiLoraBatchData) -> None:
        assert isinstance(tensor_data, torch.Tensor)
        assert batch_data is None or isinstance(batch_data, MultiLoraBatchData)

        msg_id = uuid.uuid4().int
        assert msg_id not in self.backward_cache_

        if self.is_stop_signal(tensor_data):
            msg_id = -1

        phony: torch.Tensor = SendOperator.apply(torch.tensor(
            1.0, requires_grad=True), tensor_data, self.transport_, msg_id, batch_data)

        if self.is_stop_signal(tensor_data):
            return

        self.backward_cache_[msg_id] = phony

    def init_partition(self, model: LLMModel) -> None:
        balance = self.balance_[self.rank_]
        start_module_idx = sum(self.balance_[:self.rank_])
        logging.info(
            f"RANK-{self.rank_} in device {self.device_} to load module layers from {start_module_idx} to {start_module_idx + balance}.")

        seq_model = model.sequential_module()
        del seq_model[:start_module_idx]
        del seq_model[balance:]
        for idx in range(0, len(seq_model)):
            self.model_partition_.append(seq_model[idx])

        assert len(self.model_partition_) == balance

        self.n_heads_ = model.n_heads_
        worker_train_paramas: Dict[str,
                                   List[torch.Tensor]] = model.get_train_paramas()
        self.optimizer_ = create_optimizer_from(
            self.config_, worker_train_paramas)
        self.accumulation_step_ = get_accumulation_steps(self.config_)

        del model

        torch.cuda.empty_cache()

    def forward(self,
                tensor_data: torch.Tensor,
                batch_data: MultiLoraBatchData):
        mask = precompute_mask(tensor_data, self.n_heads_,
                               self.device_, batch_data.additional_mask_)
        data = (tensor_data, mask, batch_data, True)

        for seq in self.model_partition_:
            data = seq.forward(data)

        return data[0]
