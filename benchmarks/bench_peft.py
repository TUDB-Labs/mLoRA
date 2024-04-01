from mlora.utils import setup_seed
from mlora.profiler.profiler import setup_trace_mode, grad_fn_nvtx_wrapper_by_tracepoint, set_backward_tracepoint

import torch
import random
import argparse
import logging

from transformers import LlamaForCausalLM
from peft import LoraConfig, TaskType, PeftModelForCausalLM, prepare_model_for_kbit_training

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
parser.add_argument('--peft_mode', type=str, default="seq",
                    help="How to use peft to train multi lora, include: seq, switch")

g_default_rank = 16
g_default_alpha = 16
g_default_dropout = 0.05
g_default_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
g_default_loss_fn = torch.nn.CrossEntropyLoss()

args = parser.parse_args()
assert not (args.load_4bit and args.load_8bit)


def setup_lora_adapter(llm_model: LlamaForCausalLM) -> PeftModelForCausalLM:
    peft_llm_model = llm_model

    for idx in range(0, args.lora_cnt):
        adapter_name = f"lora_{idx}"
        lora_r = g_default_rank
        lora_alpha = g_default_alpha
        lora_dropout = g_default_dropout
        lora_target = g_default_target_modules
        peft_lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                      r=lora_r,
                                      lora_alpha=lora_alpha,
                                      lora_dropout=lora_dropout,
                                      target_modules=lora_target,
                                      bias="none",
                                      inference_mode=False)
        peft_llm_model = PeftModelForCausalLM(
            peft_llm_model, peft_lora_config, adapter_name)

    return peft_llm_model


def setup_llm_model() -> LlamaForCausalLM:
    load_bits = None
    load_bits = 8 if args.load_8bit else load_bits
    load_bits = 4 if args.load_4bit else load_bits

    qlora_4bit_fp16 = True
    qlora_4bit_bf16 = False
    qlora_4bit_double_quant = True
    qlora_4_bit_quant_type = "nf4"

    additional_load_args = {
        "device_map": args.device,
        "torch_dtype": torch.float32
    }

    if load_bits is not None:
        logging.info('Loading model with quantization, bits = %i' % load_bits)
        from transformers import BitsAndBytesConfig
        qlora_4bit_compute_dtype = torch.float32
        # if set the compute type, then change it, otherwise hold the default
        qlora_4bit_compute_dtype = torch.float16 if qlora_4bit_fp16 else qlora_4bit_compute_dtype
        qlora_4bit_compute_dtype = torch.bfloat16 if qlora_4bit_bf16 else qlora_4bit_compute_dtype

        torch_dtype = torch.float32
        torch_dtype = torch.bfloat16 if qlora_4bit_bf16 else torch_dtype
        additional_load_args["torch_dtype"] = torch_dtype
        additional_load_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True if load_bits == 4 else False,
            load_in_8bit=True if load_bits == 8 else False,
            llm_int8_enable_fp32_cpu_offload=True,
            # only for qlora 4bit
            bnb_4bit_compute_dtype=qlora_4bit_compute_dtype,
            bnb_4bit_use_double_quant=qlora_4bit_double_quant,
            bnb_4bit_quant_type=qlora_4_bit_quant_type,
        )

    llm_model = LlamaForCausalLM.from_pretrained(
        args.base_model, **additional_load_args)

    llm_model = prepare_model_for_kbit_training(llm_model)
    llm_model.training = True
    llm_model.gradient_checkpointing_enable()

    return llm_model


def setup_labels() -> torch.Tensor:
    batch_input_ids = []
    for _ in range(0, args.batch_size):
        batch_input_ids.append([random.randint(1, 10000)
                               for _ in range(args.seq_len)])
    return torch.tensor(batch_input_ids, dtype=torch.long, device=args.device)


if __name__ == "__main__":
    lables = setup_labels()

    setup_seed(42)
    model: LlamaForCausalLM = setup_llm_model()
    vocab_size = model.vocab_size
    model: PeftModelForCausalLM = setup_lora_adapter(model)
    model.train()

    # to wramup
    for test_idx in range(0, args.warmup):
        loss = model.forward(input_ids=lables, labels=lables)[0]

    setup_trace_mode()

    def lora_seq():
        for lora_idx in range(0, args.lora_cnt):
            now_lora = f"lora_{lora_idx}"
            model.set_adapter(now_lora)
            for _ in range(0, args.repete):
                loss = model.forward(input_ids=lables, labels=lables)[0]
                set_backward_tracepoint(loss.grad_fn, "b_loss")
                grad_fn_nvtx_wrapper_by_tracepoint(loss.grad_fn)
                loss.backward()

    def lora_switch():
        for _ in range(0, args.repete):
            for lora_idx in range(0, args.lora_cnt):
                now_lora = f"lora_{lora_idx}"
                model.set_adapter(now_lora)
                loss = model.forward(input_ids=lables, labels=lables)[0]
                set_backward_tracepoint(loss.grad_fn, "b_loss")
                grad_fn_nvtx_wrapper_by_tracepoint(loss.grad_fn)
                loss.backward()

    mode_function = {
        "seq": lora_seq,
        "switch": lora_switch,
    }

    peft_mode = args.peft_mode

    assert peft_mode in mode_function, NotImplementedError
    mode_function[peft_mode]()
