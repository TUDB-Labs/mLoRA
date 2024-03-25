from mlora.utils import setup_seed
from mlora.config import LoraConfig
from mlora.model.modelargs import MultiLoraBatchData, LoraBatchDataConfig
from mlora.profiler.profiler import setup_trace_mode, set_backward_tracepoint, grad_fn_nvtx_wrapper_by_tracepoint, nvtx_range

import mlora
import torch
import random
import argparse

from typing import List

# Command Line Arguments
parser = argparse.ArgumentParser(description='PEFT benchmarks')
parser.add_argument('--base_model', type=str, required=True,
                    help='Path to or name of base model')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='Specify which GPU to be used, default is cuda:0')
# load quant
parser.add_argument('--load_8bit', action="store_true",
                    help='Load model in 8bit mode')
parser.add_argument('--load_4bit', action="store_true",
                    help='Load model in 4bit mode')
# lora test number
parser.add_argument('--lora_cnt', type=int, default=4,
                    help='The number of lora')
# test configure
parser.add_argument('--warmup', type=int, default=100,
                    help="The step of warm up")
parser.add_argument('--repete', type=int, default=100,
                    help="Total test iteration")
parser.add_argument('--seq_len', type=int, default=128,
                    help="The length of the sequence")
parser.add_argument('--batch_size', type=int, default=8,
                    help="The batch size of each lora input")


g_default_rank = 16
g_default_alpha = 16
g_default_dropout = 0.05
g_default_target_modules = {"q_proj": True,
                            "k_proj": True,
                            "v_proj": True,
                            "o_proj": True,
                            "w1_proj": False,
                            "w2_proj": False,
                            "w3_proj": False}
g_default_loss_fn = torch.nn.CrossEntropyLoss()

args = parser.parse_args()
assert not (args.load_4bit and args.load_8bit)


def setup_lora_adapter_config() -> List[LoraConfig]:
    lora_config: List[LoraConfig] = []

    for idx in range(0, args.lora_cnt):
        lora_config.append(LoraConfig({
            "name": f"lora_{idx}",
            "r": g_default_rank,
            "alpha": g_default_alpha,
            "dropout": g_default_dropout,
            "target_modules": g_default_target_modules,
            "batch_size": args.batch_size,
            "micro_batch_size": args.batch_size,
            # unused
            "test_batch_size": 0,
            "num_epochs": 0,
            "data": "",
            "test_data": "",
            "prompt": "",
            "group_by_length": "",
            "expand_side": "",
            "optim": "sgd",
            "momentum": 0.0,
            "lr": 0.0,
        }))

    return lora_config


def setup_input() -> MultiLoraBatchData:
    batch_tokens = []
    additional_masks = []
    lora_batch_data_config: List[LoraBatchDataConfig] = []

    start_idx = 0
    end_idx = 0

    for lora_idx in range(0, args.lora_cnt):
        adapter_name = f"lora_{lora_idx}"

        for _ in range(0, args.batch_size):
            tokens = [random.randint(1, 10000) for _ in range(args.seq_len)]
            batch_tokens.append(tokens)
            additional_masks.append([False] * args.seq_len)
            end_idx += 1

        lora_batch_data_config.append(LoraBatchDataConfig(
            adapter_name_=adapter_name,
            batch_start_idx_=start_idx,
            batch_end_idx_=end_idx,
        ))

        start_idx = end_idx

    return MultiLoraBatchData(batch_tokens_=batch_tokens,
                              additional_mask_=additional_masks,
                              lora_batch_data_config_=lora_batch_data_config,
                              inference_model_=False)


def calc_loss(train_data: MultiLoraBatchData, model_output: torch.Tensor) -> torch.Tensor:
    labels = torch.tensor(train_data.batch_tokens_, dtype=torch.long)
    total_loss = None

    for lora_config in train_data.lora_batch_data_config_:
        start_idx = lora_config.batch_start_idx_
        end_idx = lora_config.batch_end_idx_
        vocab_size = model_output.shape[-1]
        loss_input = model_output[start_idx:end_idx][...,
                                                     :-1, :].contiguous().view(-1, vocab_size)
        loss_target = labels[start_idx:end_idx][...,
                                                1:].contiguous().view(-1).to(loss_input.device)
        loss = g_default_loss_fn(loss_input, loss_target)
        if total_loss is None:
            total_loss = loss
        else:
            total_loss += loss

    return total_loss


if __name__ == "__main__":
    input_data = setup_input()

    setup_seed(42)

    _, model = mlora.load_base_model(args.base_model,
                                     "llama",
                                     args.device,
                                     args.load_4bit,
                                     args.load_8bit,
                                     None)

    mlora.init_lora_model(model, setup_lora_adapter_config())

    # to wramup
    for test_idx in range(0, args.warmup):
        output = model.forward(input_data)

    setup_trace_mode()

    for _ in range(0, args.repete):
        output = model.forward(input_data)
        with nvtx_range("f_calc_loss"):
            total_loss = calc_loss(input_data, output)
        set_backward_tracepoint(total_loss.grad_fn, "b_loss")
        grad_fn_nvtx_wrapper_by_tracepoint(total_loss.grad_fn)

        total_loss.backward()
