# ASPEN: Efficient Multi-LoRA Fine Tuning with Shared-Based Model
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright (C) 2023 All Rights Reserved.
#
# Github:  https://github.com/TUDB-Labs/multi-lora-fine-tune

import os
import json
import torch
import aspen
import datetime
import argparse
from typing import Dict, Tuple, List

# Command Line Arguments

parser = argparse.ArgumentParser(description='ASPEN main program')
parser.add_argument('--base_model', type=str,
                    help='Path to or name of base model')
parser.add_argument('--tokenizer', type=str,
                    help='Path to or name of tokenizer')
parser.add_argument('--load_8bit', type=bool, default=False,
                    help='Load model in 8bit mode')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='Specify which GPU to be used, default is cuda:0')
parser.add_argument('--config', type=str,
                    help='Path to finetune configuration')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed in integer, default is 42')
parser.add_argument('--log', type=bool, default=True,
                    help='Turn on or off log, default is true')

args = parser.parse_args()


def log(msg: str):
    if args.log:
        print('[%s] ASPEN: %s' %
              (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))


if torch.cuda.is_available():
    log('NVIDIA CUDA initialized successfully.')
    log('Total %i GPU(s) detected.' % torch.cuda.device_count())
else:
    print('ASPEN requires NVIDIA CUDA computing capacity. Please check your PyTorch installation.')
    exit(-1)


if args.base_model is None:
    print('error: Argument --base_model are required.')
    parser.print_help()
    exit(-1)


if args.config is None:
    print('error: Argument --config are required.')
    parser.print_help()
    exit(-1)


# Functions


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_base_model() -> Tuple[aspen.Tokenizer, aspen.LlamaModel]:
    if not os.path.isdir(args.base_model):
        raise "can't find the model file."

    model = aspen.LlamaModel.from_pretrained(
        path=args.base_model,
        device=args.device,
        load_in_8bit=args.load_8bit)

    if args.tokenizer:
        tokenizer = aspen.Tokenizer(args.tokenizer)
    else:
        tokenizer = aspen.Tokenizer(
            args.base_model + os.sep + 'tokenizer.model')

    tokenizer.pad_id_ = model.pad_id_

    return tokenizer, model


def init_lora_model(config: Dict[str, any], llama_model: aspen.LlamaModel):
    for lora_config in config["lora"]:
        llama_model.init_random_lora_weight(lora_config["name"],
                                            lora_config["r"],
                                            lora_config["alpha"],
                                            lora_config["dropout"],
                                            lora_config["target_modules"])


def get_optimizer(config: Dict[str, any], train_paramas: Dict[str, torch.Tensor]):
    optimizer_list: List[torch.optim.Optimizer] = []
    for lora_config in config["lora"]:
        adapter_name = lora_config["name"]
        optim_name = lora_config["optim"]
        lr = lora_config["lr"]
        if optim_name == "sgd":
            momentum = 0
            if "momentum" in lora_config:
                momentum = lora_config["momentum"]
            optimizer_list.append(torch.optim.SGD(
                train_paramas[adapter_name], lr=lr, momentum=momentum))
        elif optim_name == "adamw":
            optimizer_list.append(torch.optim.AdamW(
                train_paramas[adapter_name], lr=lr))
        else:
            raise f"unkown optimizer {optim_name}"
    return optimizer_list


def train(config: Dict[str, any], llama_model: aspen.LlamaModel, data_set: aspen.DataSet):
    train_paramas = llama_model.get_train_paramas(config)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer_list = get_optimizer(config, train_paramas)

    step_cnt = 0
    while not data_set.check_done():
        for optim in optimizer_list:
            optim.zero_grad()

        input: aspen.MultiLoraBatchData = data_set.get_batch_data()

        step_cnt += 1

        output = llama_model.forward(input)
        labels = torch.tensor(input.batch_tokens_,
                              dtype=torch.long).to(args.device)

        total_loss = None
        for lora_config in input.lora_batch_data_config_:
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_
            loss_input = output[start_idx:end_idx][..., :-1,
                                                   :].contiguous().view(-1, llama_model.vocab_size_)
            loss_target = labels[start_idx:end_idx][...,
                                                    1:].contiguous().view(-1)
            loss = loss_fn(loss_input, loss_target) / (end_idx - start_idx)
            print(
                f"    adapter: {lora_config.adapter_name_} loss: {loss}")
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        total_loss.backward()
        for optim in optimizer_list:
            optim.step()

        if step_cnt % config["save_step"] == 0:
            aspen.save_lora_model(llama_model, config, f"{step_cnt}")

    aspen.save_lora_model(llama_model, config)


# Main Function


if __name__ == "__main__":
    setup_seed(args.seed)

    with open(args.config, 'r', encoding='utf8') as fp:
        config = json.load(fp)

    tokenizer, model = load_base_model()
    init_lora_model(config, model)

    torch.cuda.empty_cache()

    data_set = aspen.DataSet(config, tokenizer)

    train(config, model, data_set)
