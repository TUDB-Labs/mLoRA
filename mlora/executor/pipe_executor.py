import logging
import time
import uuid
from enum import Enum, auto
from typing import Any, Dict, List, OrderedDict, cast

import torch

from mlora.config import MLoRAConfig
from mlora.config.task import TaskConfig
from mlora.model.args import LinearInfo, MLoRAData, ModelData
from mlora.model.llm import LLMModel
from mlora.model.llm.model_llama import precompute_mask
from mlora.model.tokenizer import Tokenizer

from .dispatcher import DISPATCHER_CLASS, PipeDispatcher
from .executor import Executor
from .pipeline.function import RecvOperator, SendOperator
from .pipeline.messages import PipeMessage, PipeMessageType
from .pipeline.queue import DeviceSwapQueue
from .pipeline.rpc_transport import RpcTransport
from .pipeline.stream import CudaStream
from .task import Task


class WorkerRole(Enum):
    HEAD = auto()
    MID = auto()
    TAIL = auto()


class PipeExecutor(Executor):
    role_: WorkerRole
    device_: str

    rank_: int
    world_size_: int
    balance_: List[int]

    # info about model
    partial_model_: torch.nn.Sequential
    heads_: int
    model_name_: str
    recompute_: bool

    input_queue_: DeviceSwapQueue
    transport_: RpcTransport

    # cache some tensor
    backward_cache_: Dict[int, torch.Tensor]
    input_cache_: Dict[int, MLoRAData]

    dispatcher_: PipeDispatcher

    def __init__(
        self,
        model: LLMModel,
        tokenizer: Tokenizer,
        config: MLoRAConfig,
        device: str,
        rank: int,
        balance: List[int],
        recompute: bool = False,
    ) -> None:
        self.model_ = model
        self.tokenizer_ = tokenizer
        self.heads_ = self.model_.n_heads_
        self.model_name_ = self.model_.name_or_path_

        self.device_ = device
        self.rank_ = rank
        self.balance_ = balance
        self.world_size_ = len(balance)

        self.backward_cache_ = {}
        self.input_cache_ = {}

        self.recompute_ = recompute

        self.__init_worker()
        self.__init_partition()

        # record the default stream to sync
        self.default_stream_ = CudaStream(torch.cuda.default_stream(self.device_))
        # init the rpc and wait the cluster node ready
        self.transport_ = RpcTransport(
            self.rank_, self.world_size_, torch.device(self.device_)
        )

        self.dispatcher_: PipeDispatcher = cast(
            PipeDispatcher, DISPATCHER_CLASS["pipe"](config.dispatcher_)
        )

        hook_func = {
            "init": self.__task_init_hook,
            "running": self.__task_to_running_hook,
            "ready": self.__task_to_ready_hook,
            "done": self.__task_to_done_hook,
            "terminate": self.__task_to_terminate_hook,
        }

        for hook, cb in hook_func.items():
            self.dispatcher_.register_hook(hook, cb)

    def __init_worker(self):
        # init the different worker
        if self.rank_ == 0:
            self.role_ = WorkerRole.HEAD
            self.input_queue_ = DeviceSwapQueue(
                torch.device("cpu"), torch.device(self.device_), 4, "input_data_queue"
            )
            self.input_queue_.start()
        elif self.rank_ == self.world_size_ - 1:
            self.role_ = WorkerRole.TAIL
        else:
            self.role_ = WorkerRole.MID

    def __init_partition(self) -> None:
        balance = self.balance_[self.rank_]
        start_module_idx = sum(self.balance_[: self.rank_])
        end_module_idx = start_module_idx + balance
        logging.info(
            f"RANK-{self.rank_} in device {self.device_} to load module layers "
            f"from {start_module_idx} to {end_module_idx}."
        )

        seq_model: torch.nn.Sequential = self.model_.sequential()
        assert sum(self.balance_) == len(seq_model)

        self.partial_model_ = torch.nn.Sequential()

        for idx in range(start_module_idx, end_module_idx):
            self.partial_model_.append(seq_model[idx])

        assert len(self.partial_model_) == balance

        del seq_model[:start_module_idx]
        del seq_model[balance:]
        del self.model_

        torch.cuda.empty_cache()

    def __head_worker_run(self):
        while True:
            # we get the model's output, and calc the loss
            self.__process_comm()
            self.__process_backward()
            self.__process_output()
            self.__process_input()
            time.sleep(1 / 100000)

    def __not_head_worker_run(self):
        while True:
            self.__process_comm()
            self.__process_backward()
            self.__process_forward()
            time.sleep(1 / 100000)

    def __head_process_step(self, message: PipeMessage):
        assert message.model_data_ is not None
        train_data: MLoRAData = self.input_cache_[message.model_data_.random_id_]

        # like dpo one task have two data config
        task_names = set()
        for item in train_data.data_config_:
            task_names.add(item.task_name_)

        for task_name in task_names:
            self.dispatcher_.task_step(task_name)
            self.dispatcher_.unlock_task(task_name)

        assert message.model_data_ is not None
        del self.input_cache_[message.model_data_.random_id_]

    def __process_backward(self):
        message = self.transport_.recv_message(PipeMessageType.GRADIENTS, block=False)
        if message is None:
            return

        logging.debug(
            f"Recv the gradients - {str(message.msg_id_)[:8]} from {message.src_}."
        )

        msg_id = message.msg_id_

        assert msg_id in self.backward_cache_

        phony: torch.Tensor = self.backward_cache_[msg_id]
        phony.grad_fn.grad_from_next_worker = message.tensor_data_  # type: ignore
        phony.backward()

        del self.backward_cache_[msg_id]

        if self.role_ == WorkerRole.HEAD:
            self.__head_process_step(message)
        else:
            assert message.model_data_ is not None
            for task_name in message.model_data_.task_name_:
                self.dispatcher_.dispatch_task_to_step(task_name)

    def __process_forward(self):
        assert self.role_ != WorkerRole.HEAD

        # recv the tensors from prev-worker
        message = self.transport_.recv_message(PipeMessageType.ACTIVATIONS, block=False)
        if message is None:
            return

        logging.debug(
            f"Recv the activations - {str(message.msg_id_)[:8]} from {message.src_}."
        )

        data = RecvOperator.apply(
            torch.tensor(1.0, requires_grad=True), self.transport_, message
        )
        # we need to wait the default stream calcuate all tensor
        # and then send it, so we hook the pre stage fn to poll the stream
        data.grad_fn.pre_stage_fn = self.default_stream_.poll  # type: ignore
        assert message.model_data_ is not None
        data = self.__forward(data, message.model_data_)

        self.default_stream_.poll()
        assert message.model_data_ is not None
        return self.__send_activations(data, message.model_data_)

    def __process_comm(self):
        try:
            msg: PipeMessage = self.transport_.recv_comm(
                PipeMessageType.COMM, block=False
            )
            comm_data = msg.comm_data_
        except Exception:
            return

        if comm_data["comm"] == "task_add":
            self.add_task(comm_data["data"])
        elif comm_data["comm"] == "task_running":
            self.dispatcher_.dispatch_task_to_run(comm_data["data"])
        elif comm_data["comm"] == "task_ready":
            self.dispatcher_.dispatch_task_to_ready(comm_data["data"])
        elif comm_data["comm"] == "task_done":
            self.dispatcher_.dispatch_task_to_done(comm_data["data"])
        elif comm_data["comm"] == "task_terminal":
            self.dispatcher_.dispatch_task_to_terminal(comm_data["data"])
        else:
            raise NotImplementedError

    def __process_output(self):
        assert self.role_ == WorkerRole.HEAD

        # recv the tensors from prev-worker
        message = self.transport_.recv_message(PipeMessageType.ACTIVATIONS, block=False)
        if message is None:
            return

        logging.debug(
            f"Recv the activations - {str(message.msg_id_)[:8]} from {message.src_}."
        )

        output: torch.Tensor = RecvOperator.apply(
            torch.tensor(1.0, requires_grad=True), self.transport_, message
        )
        # we need to wait the default stream calcuate all tensor
        # and then send it, so we hook the pre stage fn to poll the stream
        output.grad_fn.pre_stage_fn = self.default_stream_.poll  # type: ignore

        assert message.model_data_ is not None
        train_data: MLoRAData = self.input_cache_[message.model_data_.random_id_]
        labels = torch.tensor(train_data.batch_tokens_, dtype=torch.long)
        masks = torch.tensor(train_data.batch_mask_)

        total_loss: torch.Tensor | None = None

        for config in train_data.data_config_:
            loss = config.loss_fn_(output, labels, masks)
            if loss is None:
                continue
            total_loss = loss if total_loss is None else total_loss + loss

        if total_loss is not None:
            total_loss.backward()

    def __process_input(self):
        train_data: MLoRAData | None = self.dispatcher_.data()
        if train_data is None:
            return
        # step1. get the model data and execute the forward
        tensor_data = torch.tensor(
            train_data.batch_tokens_,
            dtype=torch.long,
            device=self.device_,
            requires_grad=False,
        )

        hidden_data = self.__forward(tensor_data, train_data.model_data())

        # step2. then send the hidden state to next worker
        self.default_stream_.poll()
        self.__send_activations(hidden_data, train_data.model_data())

        # step3. cache the input, we need it to calc the loss
        self.input_cache_[train_data.model_data().random_id_] = train_data

    def __send_activations(self, tensor_data: torch.Tensor, batch_data: ModelData):
        assert isinstance(tensor_data, torch.Tensor)
        assert batch_data is None or isinstance(batch_data, ModelData)

        msg_id = uuid.uuid4().int
        assert msg_id not in self.backward_cache_

        phony: torch.Tensor = SendOperator.apply(
            torch.tensor(1.0, requires_grad=True),
            tensor_data,
            self.transport_,
            msg_id,
            batch_data,
        )

        self.backward_cache_[msg_id] = phony

    def __send_comm(self, data: Any):
        self.transport_.send_comm(PipeMessageType.COMM, data)

    def __forward(self, tensor_data: torch.Tensor, batch_data: ModelData):
        mask = precompute_mask(
            tensor_data, self.heads_, self.device_, batch_data.batch_mask_
        )
        data = (tensor_data, mask, batch_data, self.recompute_)

        for seq in self.partial_model_:
            data = seq.forward(data)

        return data[0]

    def execute(self) -> None:
        if self.role_ == WorkerRole.HEAD:
            self.__head_worker_run()
        elif self.role_ == WorkerRole.MID or self.role_ == WorkerRole.TAIL:
            self.__not_head_worker_run()
        else:
            raise NotImplementedError

    def add_task(self, config: TaskConfig):
        if self.role_ != WorkerRole.TAIL:
            self.__send_comm({"comm": "task_add", "data": config})
        if self.role_ != WorkerRole.HEAD:
            # only the head worker need to load dataset
            config.dataset_ = None
        self.dispatcher_.add_task(config, self.model_name_)

    def __task_init_hook(self, task: Task):
        logging.info(
            f"Init {task.task_type()} : {task.task_name()} "
            + f"task with adapters: {task.adapter_name()}"
        )
        task.prepare(self.__linears_info(), self.tokenizer_)

    def __task_to_running_hook(self, task: Task):
        logging.info(f"Task to running, need to load adapters: {task.adapter_name()}")
        if self.role_ != WorkerRole.TAIL:
            self.__send_comm({"comm": "task_running", "data": task.task_name()})

        task.switch_device(self.device_)
        for adapter_model in task.adapter_model():
            for partial_layer in self.partial_model_:
                if partial_layer.name() != "Decoder":
                    continue
                partial_layer.wrapper_module_.load_adapter(adapter_model)

    def __task_to_ready_hook(self, task: Task):
        logging.info(f"Base model offload adapters: {task.adapter_name()}")
        if self.role_ != WorkerRole.TAIL:
            self.__send_comm({"comm": "task_ready", "data": task.task_name()})

        for adapter_name in task.adapter_name():
            for partial_layer in self.partial_model_:
                if partial_layer.name() != "Decoder":
                    continue
                partial_layer.wrapper_module_.offload_adapter(adapter_name)
        task.switch_device("cpu")

    def __task_to_done_hook(self, task: Task):
        logging.info(f"Finish and base model offload adapter - {task.adapter_name()}")
        if self.role_ != WorkerRole.TAIL:
            self.__send_comm({"comm": "task_done", "data": task.task_name()})

        task.switch_device("cpu")
        for adapter_name in task.adapter_name():
            for partial_layer in self.partial_model_:
                if partial_layer.name() != "Decoder":
                    continue
                partial_layer.wrapper_module_.offload_adapter(adapter_name)
        task.done(is_pipeline=self.rank_)

    def __task_to_terminate_hook(self, task: Task):
        logging.info(f"Task - {task.task_name()} terminate.")
        if self.role_ != WorkerRole.TAIL:
            self.__send_comm({"comm": "task_terminal", "data": task.task_name()})

        task.switch_device("cpu")
        for adapter_name in task.adapter_name():
            for partial_layer in self.partial_model_:
                if partial_layer.name() != "Decoder":
                    continue
                partial_layer.wrapper_module_.offload_adapter(adapter_name)
        task.terminate()

    def __linears_info(self) -> OrderedDict[str, LinearInfo]:
        ret_val = OrderedDict()
        for module in self.partial_model_:
            if module.name() != "Decoder":
                continue
            ret_val.update(module.wrapper_module_.linears_info())
        return ret_val
