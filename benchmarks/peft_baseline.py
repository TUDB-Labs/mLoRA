from mlora.utils import setup_seed
from mlora.tokenizer.tokenizer import Tokenizer
from mlora.dispatcher.dispatcher import Dispatcher

import json
import torch
import argparse
import logging

from transformers import LlamaForCausalLM
from peft import LoraConfig, TaskType, PeftModelForCausalLM, prepare_model_for_kbit_training
from typing import Dict

# Command Line Arguments
parser = argparse.ArgumentParser(description='PEFT benchmarks')
parser.add_argument('--base_model', type=str, required=True,
                    help='Path to or name of base model')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='Specify which GPU to be used, default is cuda:0')
# configuration
parser.add_argument('--config', type=str, required=True,
                    help='Path to finetune configuration')
# load quant
parser.add_argument('--load_8bit', action="store_true",
                    help='Load model in 8bit mode')
parser.add_argument('--load_4bit', action="store_true",
                    help='Load model in 4bit mode')

args = parser.parse_args()
assert not (args.load_4bit and args.load_8bit)

with open(args.config, 'r', encoding='utf8') as fp:
    config = json.load(fp)


def setup_lora_adapter(llm_model: LlamaForCausalLM) -> PeftModelForCausalLM:
    peft_llm_model = llm_model

    for lora_config in config["lora"]:
        adapter_name = lora_config["name"]
        lora_r = lora_config["r"]
        lora_alpha = lora_config["alpha"]
        lora_dropout = lora_config["dropout"]
        lora_target = lora_config["target_modules"]
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
        additional_load_args["load_in_4bit"] = True if load_bits == 4 else False
        additional_load_args["load_in_8bit"] = True if load_bits == 8 else False
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

    return llm_model


def get_accumulation_steps(config: Dict[str, any]) -> Dict[str, int]:
    ret_accumulation_step = {}
    for lora_config in config["lora"]:
        batch_size = lora_config["batch_size"]
        micro_batch_size = lora_config["micro_batch_size"]
        if batch_size < micro_batch_size or batch_size % micro_batch_size != 0:
            raise f"error batch_size {batch_size} and micro batch size {micro_batch_size}"
        ret_accumulation_step[lora_config["name"]
                              ] = batch_size / micro_batch_size
    return ret_accumulation_step


def get_optimizer(model: PeftModelForCausalLM, config: Dict[str, any]) -> Dict[str, torch.optim.Optimizer]:
    # get optimizer per lora model
    optimizer: Dict[str, torch.optim.Optimizer] = {}

    for lora_config in config["lora"]:
        adapter_name = lora_config["name"]
        optim_name = lora_config["optim"]
        lr = lora_config["lr"]
        model.set_adapter(adapter_name)
        train_paramas = [p for p in model.parameters() if p.requires_grad]
        if optim_name == "sgd":
            momentum = 0
            if "momentum" in lora_config:
                momentum = lora_config["momentum"]
            optimizer[adapter_name] = (torch.optim.SGD(
                train_paramas, lr=lr, momentum=momentum))
        elif optim_name == "adamw":
            optimizer[adapter_name] = (torch.optim.AdamW(train_paramas, lr=lr))
        else:
            raise f"unkown optimizer {optim_name}"

    return optimizer


if __name__ == "__main__":
    setup_seed(42)
    model: LlamaForCausalLM = setup_llm_model()
    vocab_size = model.vocab_size
    model: PeftModelForCausalLM = setup_lora_adapter(model)
    model.train()

    tokenizer = Tokenizer(args.base_model)
    dispatcher = Dispatcher(config, tokenizer)
    dispatcher.train_lora_simultaneously_num_ = 1
    dispatcher.train_lora_candidate_num_ = 1

    all_optimizer: Dict[str, torch.optim.Optimizer] = get_optimizer(
        model, config)
    accumulation_step: Dict[str, int] = get_accumulation_steps(config)
    accumulation_cnt: Dict[str, int] = {}

    loss_fn = torch.nn.CrossEntropyLoss()

    step_cnt = 0
    for data in dispatcher.train_data():
        assert len(data.lora_batch_data_config_) == 1
        adapter_name = data.lora_batch_data_config_[0].adapter_name_

        model.set_adapter(adapter_name)
        output = model.forward(input_ids=torch.tensor(data.batch_tokens_, dtype=torch.long, device=args.device),
                               attention_mask=torch.tensor(data.additional_mask_, dtype=torch.long, device=args.device))
        labels = torch.tensor(data.batch_tokens_,
                              dtype=torch.long, device=args.device)
        loss_input = output[0][..., :-1, :].contiguous().view(-1, vocab_size)
        loss_target = labels[..., 1:].contiguous().view(-1)

        loss = loss_fn(loss_input, loss_target)
        loss.backward()

        if adapter_name not in accumulation_cnt:
            accumulation_cnt[adapter_name] = 0

        accumulation_cnt[adapter_name] += 1

        if accumulation_cnt[adapter_name] % accumulation_step[adapter_name] == 0:
            all_optimizer[adapter_name].step()
            all_optimizer[adapter_name].zero_grad()

    for adapter_name in all_optimizer:
        all_optimizer[adapter_name].step()
        all_optimizer[adapter_name].zero_grad()
