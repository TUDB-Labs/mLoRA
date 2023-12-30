# m-LoRA: Efficient Multi-LoRA Fine Tuning with Shared-Based Model
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
import mlora
import random
import datetime
import argparse
from typing import Dict, Tuple, List

# Command Line Arguments
parser = argparse.ArgumentParser(description='m-LoRA main program')
parser.add_argument('--base_model', type=str,
                    help='Path to or name of base model')
parser.add_argument('--model_type', type=str, default="llama",
                    help='The model type, support: llama, chatglm')
parser.add_argument('--inference', action="store_true",
                    help='The inference mode (just for test)')
parser.add_argument('--load_lora', action="store_true",
                    help="[Legacy] Load lora from file instead of init randomly")
parser.add_argument('--load_adapter', action="store_true",
                    help="Load adapter from file instead of init randomly")
parser.add_argument('--disable_lora', action="store_true",
                    help="[Legacy] Disable the lora modules")
parser.add_argument('--disable_adapter', action="store_true",
                    help="Disable the adapter modules")
parser.add_argument('--tokenizer', type=str,
                    help='Path to or name of tokenizer')
parser.add_argument('--load_8bit', action="store_true",
                    help='Load model in 8bit mode')
parser.add_argument('--load_4bit', action="store_true",
                    help='Load model in 4bit mode')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='Specify which GPU to be used, default is cuda:0')
parser.add_argument('--config', type=str,
                    help='Path to finetune configuration')
parser.add_argument('--output', type=str, default=".",
                    help='Path to output checkpoints')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed in integer, default is 42')
parser.add_argument('--log', type=bool, default=True,
                    help='Turn on or off log, default is true')

args = parser.parse_args()

if args.inference:
    args.load_adapter = True

# Processing legacy arguments
if args.load_lora:
    args.load_adapter = True

if args.disable_lora:
    args.disable_adapter = True


def log(msg: str):
    if args.log:
        print('[%s] m-LoRA: %s' %
              (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), msg))


if torch.cuda.is_available():
    log('NVIDIA CUDA initialized successfully.')
    log('Total %i GPU(s) detected.' % torch.cuda.device_count())
else:
    print('m-LoRA requires NVIDIA CUDA computing capacity. Please check your PyTorch installation.')
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
    random.seed(seed)


def load_base_model(config: Dict[str, any]) -> Tuple[mlora.Tokenizer, mlora.LLMModel]:
    if args.model_type == "llama":
        log("Initializing LLaMA model.")
        model = mlora.LlamaModel.from_pretrained(
            path=args.base_model,
            device=args.device,
            bits=(8 if args.load_8bit else (
                4 if args.load_4bit else None)),
            log_fn=log
        )
    elif args.model_type == "chatglm":
        log("Initializing ChatGLM model.")
        model = mlora.ChatGLMModel.from_pretrained(
            path=args.base_model,
            device=args.device,
            bits=(8 if args.load_8bit else (4 if args.load_4bit else None)),
            log_fn=log
        )
    else:
        raise f"unkown model type {args.model_type}"

    tokenizer = mlora.Tokenizer(args.base_model)

    model.pad_token_id_ = tokenizer.pad_id_

    return tokenizer, model


def init_lora_model(config: Dict[str, any], llm_model: mlora.LLMModel) -> Dict[str, mlora.LoraConfig]:
    if args.disable_adapter:
        return {"DEFAULT": mlora.LoraConfig(adapter_name_="DEFAULT")}

    config_dict = {}

    for lora_config in config["lora"]:
        lora_weight = None
        if "routing_strategy" in lora_config:
            config_class = mlora.MixConfig()
            config_class.routing_strategy_ = lora_config["routing_strategy"]
            config_class.router_aux_loss_coef_ = lora_config.get(
                "router_aux_loss_coef", 0.001)
            config_class.num_experts_ = lora_config.get("experts", 8)
            if config_class.routing_strategy_ == "basic":
                config_class.top_k_ = lora_config.get("topk", 2)
            elif config_class.routing_strategy_ == "switch":
                config_class.router_z_loss_coef_ = lora_config.get(
                    "router_z_loss_coef", 0.001)
                config_class.expert_capacity_ = lora_config.get(
                    "expert_capacity", 64)
                config_class.jitter_noise_ = lora_config.get(
                    "jitter_noise", 0.01)

            lora_file_name = "mixlora_model.bin"
        else:
            config_class = mlora.LoraConfig()
            lora_file_name = "adapter_model.bin"

        config_class.adapter_name_ = lora_config["name"]
        config_class.device_ = args.device
        config_class.lora_r_ = lora_config["r"]
        config_class.lora_alpha_ = lora_config["alpha"]
        config_class.lora_dropout_ = lora_config["dropout"]
        config_class.target_modules_ = lora_config["target_modules"]

        config_dict[config_class.adapter_name_] = config_class

        if args.load_adapter:
            adapter_file_path = lora_config["output"] + os.sep + lora_file_name
            log(f"Load adapter: {adapter_file_path}")
            lora_weight = torch.load(adapter_file_path)

        llm_model.init_lora_layer_weight(config_class, lora_weight)

    return config_dict


def get_optimizer(config: Dict[str, any], train_paramas: Dict[str, torch.Tensor]) -> Dict[str, torch.optim.Optimizer]:
    # get optimizer per lora model
    optimizer: Dict[str, torch.optim.Optimizer] = {}

    for lora_config in config["lora"]:
        adapter_name = lora_config["name"]
        optim_name = lora_config["optim"]
        lr = lora_config["lr"]
        if optim_name == "sgd":
            momentum = 0
            if "momentum" in lora_config:
                momentum = lora_config["momentum"]
            optimizer[adapter_name] = (torch.optim.SGD(
                train_paramas[adapter_name], lr=lr, momentum=momentum))
        elif optim_name == "adamw":
            optimizer[adapter_name] = (torch.optim.AdamW(
                train_paramas[adapter_name], lr=lr))
        else:
            raise f"unkown optimizer {optim_name}"

    return optimizer


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


def get_router_loss_function(config:  Dict[str, any]) -> Dict[str, torch.nn.Module]:
    loss_functions: Dict[str, torch.nn.Module] = {}
    for lora_config in config["lora"]:
        routing_strategy = lora_config.get("routing_strategy", "none")
        if routing_strategy == "basic":
            loss_functions[lora_config["name"]] = mlora.BasicRouterLoss(
                lora_config.get("router_aux_loss_coef", 0.001),
                lora_config["experts"],
                lora_config.get("topk", 2))
        elif routing_strategy == "switch":
            loss_functions[lora_config["name"]] = mlora.SwitchRouterLoss(
                lora_config.get("router_z_loss_coef", 0.001),
                lora_config.get("router_aux_loss_coef", 0.001))

    return loss_functions


# to get test result and want early stop it
def train(config: Dict[str, any], llm_model: mlora.LLMModel, dispatcher: mlora.Dispatcher):
    # the train paramas per lora model
    all_train_paramas: Dict[str, List[torch.Tensor]
                            ] = llm_model.get_train_paramas()
    all_optimizer: Dict[str, torch.optim.Optimizer] = get_optimizer(
        config, all_train_paramas)
    accumulation_step: Dict[str, int] = get_accumulation_steps(config)

    loss_fn = torch.nn.CrossEntropyLoss()
    router_loss_fn = get_router_loss_function(config)

    step_cnt = 0
    while not dispatcher.check_task_done():
        input: mlora.MultiLoraBatchData = dispatcher.get_train_data()
        for lora in input.lora_batch_data_config_:
            all_optimizer[lora.adapter_name_].zero_grad()

        step_cnt += 1

        output, router_outputs = llm_model.forward(
            input, output_router_logits=(len(router_loss_fn) > 0))

        labels = torch.tensor(input.batch_tokens_,
                              dtype=torch.long).to(args.device)

        total_loss = None
        for idx, lora_config in enumerate(input.lora_batch_data_config_):
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_
            loss_input = output[start_idx:end_idx][..., :-1,
                                                   :].contiguous().view(-1, llm_model.vocab_size_)
            loss_target = labels[start_idx:end_idx][...,
                                                    1:].contiguous().view(-1)
            loss = loss_fn(loss_input, loss_target) / \
                accumulation_step[lora_config.adapter_name_]
            if router_outputs is not None and len(router_outputs[idx]) > 0:
                router_loss = router_loss_fn[lora_config.adapter_name_](
                    router_outputs[idx])
                loss += router_loss
                print(f"    adapter: {lora_config.adapter_name_} loss: {loss}")
                print(
                    f"{' '*(6 + len(lora_config.adapter_name_))} router loss: {router_loss}")
            else:
                print(f"    adapter: {lora_config.adapter_name_} loss: {loss}")
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        total_loss.backward()
        for lora in input.lora_batch_data_config_:
            if step_cnt % accumulation_step[lora.adapter_name_] == 0:
                all_optimizer[lora.adapter_name_].step()

        if step_cnt % config["save_step"] == 0:
            llm_model.save_adapter_weight(args.output, f"{step_cnt}")

    llm_model.save_adapter_weight(args.output)


def inference_callback(cur_pos, outputs):
    print(f"POSITION: {cur_pos}")
    for adapter_name, output in outputs.items():
        print(f"{adapter_name} OUTPUT: {output[0]}")


def inference(config: Dict[str, mlora.LoraConfig],
              llm_model: mlora.LLMModel,
              tokenizer: mlora.Tokenizer):
    gen_configs: List[mlora.GenerateConfig] = []
    for _, lora_config in config.items():
        gen_configs.append(mlora.GenerateConfig(lora_config_=lora_config))

    while True:
        input_raw = input("INPUT WITHOUT PROMPT: ")
        if input_raw == "QUIT":
            return
        for config in gen_configs:
            config.prompts_ = [input_raw]
        outputs = mlora.generate(llm_model, tokenizer, gen_configs,
                                 device=args.device,
                                 stream_callback=inference_callback)
        for adapter_name, output in outputs.items():
            print(f"{adapter_name} OUTPUT IS:")
            print(output[0])


# Main Function
if __name__ == "__main__":
    setup_seed(args.seed)

    with open(args.config, 'r', encoding='utf8') as fp:
        config = json.load(fp)

    tokenizer, model = load_base_model(config)
    adapters = init_lora_model(config, model)

    torch.cuda.empty_cache()

    if args.inference:
        inference(adapters, model, tokenizer)
    else:
        dispatcher = mlora.Dispatcher(config, tokenizer)
        train(config, model, dispatcher)
