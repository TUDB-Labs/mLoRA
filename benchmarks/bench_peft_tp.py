import argparse
import os
import random
import time
from dataclasses import dataclass

import torch
from peft import LoraConfig, PeftModelForCausalLM, TaskType, get_peft_model
from torch.distributed._tensor import DeviceMesh, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from transformers import LlamaForCausalLM


@dataclass
class BenchmarkArgs:
    batch_size: int = 24
    seq_len: int = 512
    test_steps: int = 10


@dataclass
class PeftArgs:
    rank = 16
    alpha = 16
    dropout = 0.05
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]


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


def init_args():
    parser = argparse.ArgumentParser(description="PEFT TP benchmarks")
    parser.add_argument(
        "--base_model", type=str, required=True, help="Path to or name of base model"
    )
    return parser.parse_args()


def dummy_train_labels() -> torch.Tensor:
    batch_input_ids = []
    for _ in range(0, BenchmarkArgs.batch_size):
        batch_input_ids.append(
            [random.randint(1, 10000) for _ in range(BenchmarkArgs.seq_len)]
        )
    return torch.tensor(
        batch_input_ids,
        dtype=torch.long,
        device=torch.device(torch.cuda.current_device()),
    )


if __name__ == "__main__":
    args = init_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print("To init the device mesh")
    tp_mesh = DeviceMesh(device_type="cuda", mesh=[0, 1, 2, 3])

    print("To load the llama model")
    model: LlamaForCausalLM = LlamaForCausalLM.from_pretrained(
        args.base_model, use_cache=False, torch_dtype=torch.float32
    )
    for param in model.parameters():
        param.requires_grad_(False)

    model.lm_head = model.lm_head.to(torch.cuda.current_device())
    model.model.embed_tokens = model.model.embed_tokens.to(torch.cuda.current_device())
    model.model.norm = model.model.norm.to(torch.cuda.current_device())

    print("To wrapper the model")

    for layer in model.model.layers:
        layer_parallelize_plan = {
            "self_attn.q_proj": ColwiseParallel(),
            "self_attn.k_proj": ColwiseParallel(),
            "self_attn.v_proj": ColwiseParallel(),
            "self_attn.o_proj": RowwiseParallel(),
            "mlp.gate_proj": ColwiseParallel(),
            "mlp.up_proj": ColwiseParallel(),
            "mlp.down_proj": RowwiseParallel(),
        }

        layer.input_layernorm = layer.input_layernorm.to(torch.cuda.current_device())
        layer.post_attention_layernorm = layer.post_attention_layernorm.to(
            torch.cuda.current_device()
        )
        layer.self_attn.rotary_emb = layer.self_attn.rotary_emb.to(
            torch.cuda.current_device()
        )

        layer.self_attn.num_heads = model.model.config.num_attention_heads // world_size
        layer.self_attn.num_key_value_heads = (
            model.model.config.num_key_value_heads // world_size
        )
        layer.self_attn.hidden_size = model.model.config.hidden_size // world_size

        parallelize_module(layer, tp_mesh, layer_parallelize_plan)

    print("To wrapper peft model")
    peft_args = PeftArgs()
    model: PeftModelForCausalLM = setup_lora_adapter(model, peft_args)
    model.print_trainable_parameters()

    for layer in model.base_model.model.model.layers:
        layer_parallelize_plan = {
            "self_attn.q_proj.lora_A.default": RowwiseParallel(
                input_layouts=Replicate()
            ),
            "self_attn.k_proj.lora_A.default": RowwiseParallel(
                input_layouts=Replicate()
            ),
            "self_attn.v_proj.lora_A.default": RowwiseParallel(
                input_layouts=Replicate()
            ),
            "self_attn.o_proj.lora_A.default": ColwiseParallel(input_layouts=Shard(-1)),
            "self_attn.q_proj.lora_B.default": ColwiseParallel(),
            "self_attn.k_proj.lora_B.default": ColwiseParallel(),
            "self_attn.v_proj.lora_B.default": ColwiseParallel(),
            "self_attn.o_proj.lora_B.default": RowwiseParallel(),
        }
        parallelize_module(layer, tp_mesh, layer_parallelize_plan)

    optimizer = torch.optim.AdamW(model.parameters())

    start_time = time.time()
    total_tokens = 0

    for step in range(BenchmarkArgs.test_steps):
        data = dummy_train_labels().to(local_rank)
        total_tokens += data.numel()

        loss = model(input_ids=data, labels=data).loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        print(f"average {total_tokens / (time.time() - start_time) : .2f} tokens/s")
