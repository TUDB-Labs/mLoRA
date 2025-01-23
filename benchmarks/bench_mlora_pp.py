import logging
import random
import time
from typing import List, Tuple, override

import mlora.config
import mlora.executor
import mlora.executor.dispatcher
import mlora.model
import mlora.utils
import torch
from mlora.config.adapter import LoRAConfig
from mlora.config.dispatcher import DispatcherConfig
from mlora.config.task import TrainTaskConfig
from mlora.executor.task import TrainTask, register_task_class
from mlora.model.args import MLoRADataConfig, Tokens

g_start_time: int | None = None
g_total_token: int = 0


class BenchmarkArgs:
    batch_size_: int = 8
    seq_len_: int = 512
    concurrency_num_: int = 4
    test_epochs_: int = 20


class BenchmarkConfig(mlora.config.MLoRAConfig):
    dispatcher_: DispatcherConfig

    def __init__(self):
        # just for init
        self.dispatcher_ = DispatcherConfig(
            {"name": "pipe", "concurrency_num": BenchmarkArgs.concurrency_num_}
        )


class BenchmarkTask(TrainTask):
    def __init__(self, config: mlora.config.TaskConfig, llm_name: str) -> None:
        super().__init__(config, llm_name)

    @override
    def data(self, start_idx: int) -> Tuple[List[Tokens], List[MLoRADataConfig]]:

        ret_tokens = [
            [random.randint(1, 10000) for _ in range(BenchmarkArgs.seq_len_)]
        ] * BenchmarkArgs.batch_size_

        end_idx = start_idx + len(ret_tokens)

        def loss_fn(
            input: torch.Tensor, target: torch.Tensor, _: torch.Tensor
        ) -> torch.Tensor:
            vacab_size = input.shape[-1]
            loss_input = (
                input[start_idx:end_idx, :-1, :].contiguous().view(-1, vacab_size)
            )
            loss_target = (
                target[start_idx:end_idx, 1:]
                .contiguous()
                .view(-1)
                .to(loss_input.device)
            )
            loss: torch.Tensor = self.context_.loss_fn_(loss_input, loss_target)

            return loss

        data_config = MLoRADataConfig(
            self.context_.name_,
            self.context_.type_,
            start_idx,
            end_idx,
            self._expand_batch_tokens,
            loss_fn,
            self.task_name(),
        )

        return ret_tokens, [data_config]

    @override
    def step(self):
        self.now_step_ += 1
        self.now_epoch_ += 1
        self.context_.step()

        global g_start_time
        global g_total_token
        if g_start_time is not None:
            g_total_token = g_total_token + (
                BenchmarkArgs.batch_size_ * BenchmarkArgs.seq_len_
            )
            logging.info(
                f"average {g_total_token / (time.time() - g_start_time) : .2f} tokens/s"
            )
        else:
            g_start_time = time.time()

        logging.info(f"task {self.context_.name_} step")


def generate_task_config(task_idx: int) -> TrainTaskConfig:
    adapters = {
        f"test_{task_idx}": LoRAConfig(
            {
                "type": "lora",
                "name": f"test_{task_idx}",
                "path": f"adapters/test_{task_idx}",
                "r": 16,
                "alpha": 16,
                "dropout": 0.05,
                "target_modules": {
                    "q_proj": True,
                    "k_proj": True,
                    "v_proj": True,
                    "o_proj": True,
                },
                "optimizer": "adamw",
                "lr": 1e-3,
            }
        )
    }
    datasets = {f"test_{task_idx}": None}

    return TrainTaskConfig(
        {
            "batch_size": BenchmarkArgs.batch_size_,
            "mini_batch_size": BenchmarkArgs.batch_size_,
            "num_epochs": BenchmarkArgs.test_epochs_,
            "cutoff_len": BenchmarkArgs.seq_len_,
            "save_step": 10000000,
            "name": f"test_{task_idx}",
            "type": "benchmark",
            "adapter": f"test_{task_idx}",
            "dataset": f"test_{task_idx}",
        },
        adapters,
        datasets,
    )


if __name__ == "__main__":
    args = mlora.utils.get_cmd_args()

    mlora.utils.setup_seed(args.seed)
    mlora.utils.setup_logging(args.log_level, args.log_file)
    mlora.utils.setup_cuda_check()

    register_task_class("benchmark", BenchmarkTask)

    # enable the trace mode for profiling performance
    if args.trace:
        mlora.utils.setup_trace_mode()

    tokenizer, model = mlora.model.load_model(args)

    config = BenchmarkConfig()

    # init all task from config file
    executor = mlora.executor.PipeExecutor(
        model, tokenizer, config, args.device, args.rank, args.balance, args.recompute
    )

    # only the header node can add task
    if args.rank == 0:
        for idx in range(0, BenchmarkArgs.concurrency_num_):
            executor.add_task(generate_task_config(idx))

    executor.execute()
