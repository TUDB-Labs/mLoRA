import os
import fire
import random
import torch
import torch.optim as optim
import time
import argparse

from peft import get_peft_model, PeftModelForCausalLM, LoraConfig, TaskType
from dataclasses import dataclass
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from llama_recipes.utils.fsdp_utils import fsdp_auto_wrap_policy
from llama_recipes.configs.fsdp import fsdp_config as FSDP_CONFIG
from llama_recipes.configs.training import train_config as TRAIN_CONFIG
from llama_recipes.policies import apply_fsdp_checkpointing
from llama_recipes.utils.train_utils import (
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)


@dataclass
class BenchmarkArgs():
    batch_size: int = 8
    seq_len: int = 1024
    accumulation_steps: int = 4
    test_steps: int = 100


@dataclass
class PeftArgs():
    rank = 16
    alpha = 16
    dropout = 0.05
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


def setup_labels(benchmark_args: BenchmarkArgs) -> torch.Tensor:
    batch_input_ids = []
    for _ in range(0, benchmark_args.batch_size):
        batch_input_ids.append([random.randint(1, 10000)
                               for _ in range(benchmark_args.seq_len)])
    return torch.tensor(batch_input_ids, dtype=torch.long)


def create_optimizer(model: FSDP, train_config: TRAIN_CONFIG):
    optimizer = optim.AdamW(model.parameters(),
                            lr=train_config.lr,
                            weight_decay=train_config.weight_decay)
    return optimizer


def setup_lora_adapter(model: LlamaForCausalLM, peft_args: PeftArgs) -> PeftModelForCausalLM:
    peft_lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                  r=peft_args.rank,
                                  lora_alpha=peft_args.alpha,
                                  lora_dropout=peft_args.dropout,
                                  target_modules=peft_args.target_modules,
                                  bias="none",
                                  inference_mode=False)
    model = get_peft_model(model, peft_lora_config)

    return model


def create_model(rank: int, fsdp_config: FSDP_CONFIG, train_config: TRAIN_CONFIG) -> FSDP:
    model = LlamaForCausalLM.from_pretrained(
        train_config.model_name,
        use_cache=False
    )
    print_model_size(model, train_config, rank)

    # setup lora
    peft_args = PeftArgs()
    model = setup_lora_adapter(model, peft_args)
    model.print_trainable_parameters()

    hsdp_device_mesh = None
    if fsdp_config.hsdp and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
        hsdp_device_mesh = hsdp_device_mesh(replica_group_size=fsdp_config.replica_group_size,
                                            sharding_group_size=fsdp_config.sharding_group_size)
        print("HSDP device mesh is ready")

    # setup FSDP
    device_id = torch.cuda.current_device()
    my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)
    mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
    model = FSDP(model,
                 auto_wrap_policy=my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
                 cpu_offload=None,
                 mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
                 sharding_strategy=fsdp_config.sharding_strategy,
                 device_mesh=hsdp_device_mesh,
                 device_id=device_id,
                 limit_all_gathers=True,
                 sync_module_states=train_config.low_cpu_fsdp,
                 param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
                 if train_config.low_cpu_fsdp and rank != 0 else None,)
    if fsdp_config.fsdp_activation_checkpointing:
        apply_fsdp_checkpointing(model)
    return model


def init_args():
    parser = argparse.ArgumentParser(description='PEFT FSDP benchmarks')
    parser.add_argument('--base_model', type=str, required=True,
                        help='Path to or name of base model')
    return parser.parse_args()


def main():
    args = init_args()

    setup()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)
    setup_environ_flags(rank)

    fsdp_config = FSDP_CONFIG()

    train_config = TRAIN_CONFIG()
    train_config.model_name = args.base_model
    train_config.enable_fsdp = True
    train_config.use_peft = True

    benchmark_args = BenchmarkArgs()

    model = create_model(rank, fsdp_config, train_config)
    optimizer = create_optimizer(model, train_config)
    labels = setup_labels(benchmark_args)

    train(model, optimizer, labels, local_rank, benchmark_args)


def train(model: FSDP,
          optimizer: optim.AdamW,
          labels: torch.Tensor,
          local_rank: int,
          benchmark_args: BenchmarkArgs):
    autocast = torch.cuda.amp.autocast
    start_time = time.time()
    total_tokens = 0
    for step in range(benchmark_args.test_steps):
        data = labels.to(local_rank)
        total_tokens += data.numel()
        with autocast():
            loss = model(input_ids=data, labels=data).loss
        loss.backward()

        if (step + 1) % benchmark_args.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        if local_rank == 0:
            print(f'average {total_tokens / (time.time() - start_time) : .2f} tokens/s')


if __name__ == "__main__":
    fire.Fire(main)
