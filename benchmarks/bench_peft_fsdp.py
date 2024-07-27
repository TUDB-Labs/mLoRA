import argparse
import os
import random
import time
from dataclasses import dataclass

import llama_recipes
import llama_recipes.configs
import llama_recipes.utils.fsdp_utils
import llama_recipes.utils.train_utils
import torch
import torch.distributed.fsdp
import torch.optim as optim
from llama_recipes.policies import apply_fsdp_checkpointing
from peft import LoraConfig, PeftModelForCausalLM, TaskType, get_peft_model
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


@dataclass
class BenchmarkArgs:
    batch_size: int = 8
    seq_len: int = 512
    test_steps: int = 100


@dataclass
class PeftArgs:
    rank = 16
    alpha = 16
    dropout = 0.05
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


def dummy_train_labels(benchmark_args: BenchmarkArgs) -> torch.Tensor:
    batch_input_ids = []
    for _ in range(0, benchmark_args.batch_size):
        batch_input_ids.append(
            [random.randint(1, 10000) for _ in range(benchmark_args.seq_len)]
        )
    return torch.tensor(batch_input_ids, dtype=torch.long)


def create_optimizer(
    model: torch.distributed.fsdp.FullyShardedDataParallel,
    train_config: llama_recipes.configs.train_config,
):
    optimizer = optim.AdamW(
        model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
    )
    return optimizer


def setup_lora_adapter(
    model: LlamaForCausalLM, peft_args: PeftArgs
) -> PeftModelForCausalLM:
    peft_lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=peft_args.rank,
        lora_alpha=peft_args.alpha,
        lora_dropout=peft_args.dropout,
        target_modules=peft_args.target_modules,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, peft_lora_config)

    return model


def create_peft_model(
    rank: int,
    fsdp_config: llama_recipes.configs.fsdp_config,
    train_config: llama_recipes.configs.train_config,
) -> torch.distributed.fsdp.FullyShardedDataParallel:
    # init the backbone and lora adapter
    model = LlamaForCausalLM.from_pretrained(
        train_config.model_name, use_cache=False, torch_dtype=torch.float32
    )

    peft_args = PeftArgs()
    model = setup_lora_adapter(model, peft_args)
    model.print_trainable_parameters()

    # setup FSDP
    device_id = torch.cuda.current_device()

    auto_wrapping_policy = llama_recipes.utils.fsdp_utils.fsdp_auto_wrap_policy(
        model, LlamaDecoderLayer
    )
    mixed_precision_policy, wrapping_policy = (
        llama_recipes.utils.train_utils.get_policies(fsdp_config, rank)
    )

    model = torch.distributed.fsdp.FullyShardedDataParallel(
        model,
        auto_wrap_policy=(
            auto_wrapping_policy if train_config.use_peft else wrapping_policy
        ),
        cpu_offload=None,
        mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
        sharding_strategy=fsdp_config.sharding_strategy,
        device_id=device_id,
        limit_all_gathers=True,
        sync_module_states=train_config.low_cpu_fsdp,
        param_init_fn=lambda module: (
            module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0
            else None
        ),
    )

    if fsdp_config.fsdp_activation_checkpointing:
        apply_fsdp_checkpointing(model)

    return model


def init_args():
    parser = argparse.ArgumentParser(description="PEFT FSDP benchmarks")
    parser.add_argument(
        "--base_model", type=str, required=True, help="Path to or name of base model"
    )
    return parser.parse_args()


def train(
    model: torch.distributed.fsdp.FullyShardedDataParallel,
    optimizer: optim.AdamW,
    labels: torch.Tensor,
    rank: int,
    local_rank: int,
    benchmark_args: BenchmarkArgs,
):
    start_time = time.time()
    total_tokens = 0

    for _ in range(benchmark_args.test_steps):
        data = labels.to(local_rank)
        total_tokens += data.numel()

        # with torch.cuda.amp.autocast():
        loss = model(input_ids=data, labels=data).loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if rank == 0:
            print(f"average {total_tokens / (time.time() - start_time) : .2f} tokens/s")


if __name__ == "__main__":
    args = init_args()

    llama_recipes.utils.train_utils.setup()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])

    torch.cuda.set_device(local_rank)

    llama_recipes.utils.train_utils.clear_gpu_cache(local_rank)
    llama_recipes.utils.train_utils.setup_environ_flags(rank)

    fsdp_config = llama_recipes.configs.fsdp_config(
        fsdp_activation_checkpointing=False, mixed_precision=False
    )
    train_config = llama_recipes.configs.train_config(
        model_name=args.base_model, enable_fsdp=True, use_peft=True
    )

    fsdp_model = create_peft_model(rank, fsdp_config, train_config)
    optimizer = create_optimizer(fsdp_model, train_config)

    benchmark_args = BenchmarkArgs()
    labels = dummy_train_labels(benchmark_args)

    train(fsdp_model, optimizer, labels, rank, local_rank, benchmark_args)
