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
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed in integer, default is 42')
parser.add_argument('--dir', type=str, default=".",
                    help='Path to read or save checkpoints')
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

    tokenizer = mlora.Tokenizer(args.base_model, device=args.device)

    model.pad_token_id_ = tokenizer.pad_id_

    return tokenizer, model


def init_lora_model(config: Dict[str, any], llm_model: mlora.LLMModel) -> Dict[str, mlora.LoraConfig]:
    if args.disable_adapter:
        return {"DEFAULT": mlora.LoraConfig(
            adapter_name_="DEFAULT",
            prompt_template_="template/template_demo.json",
            device_=args.device,
        )}

    config_dict = {}

    for lora_config in config["lora"]:
        lora_weight = None
        config_class = mlora.lora_config_factory(lora_config)
        config_class.adapter_name_ = lora_config["name"]
        config_class.prompt_template_ = lora_config["prompt"]
        config_class.device_ = args.device

        config_dict[config_class.adapter_name_] = config_class

        if args.load_adapter:
            adapter_file_path = args.dir + os.sep + \
                config_class.adapter_name_ + os.sep + "adapter_model.bin"
            log(f"Load adapter: {adapter_file_path}")
            lora_weight = torch.load(
                adapter_file_path, map_location=args.device)

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


def get_router_loss_function(adapters: Dict[str, mlora.LoraConfig]) -> Dict[str, torch.nn.Module]:
    loss_functions: Dict[str, torch.nn.Module] = {}
    for lora_name, lora_config in adapters.items():
        if isinstance(lora_config, mlora.MixConfig):
            loss_functions[lora_name] = mlora.router_loss_factory(lora_config)

    return loss_functions


# to get test result and want early stop it
def train(adapters: Dict[str, mlora.LoraConfig], config: Dict[str, any], llm_model: mlora.LLMModel, dispatcher: mlora.Dispatcher):
    # the train paramas per lora model
    all_train_paramas: Dict[str, List[torch.Tensor]
                            ] = llm_model.get_train_paramas()
    all_optimizer: Dict[str, torch.optim.Optimizer] = get_optimizer(
        config, all_train_paramas)
    accumulation_step: Dict[str, int] = get_accumulation_steps(config)

    loss_fn = torch.nn.CrossEntropyLoss()
    router_loss_fn = get_router_loss_function(adapters)

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
                    router_outputs[idx]) / accumulation_step[lora_config.adapter_name_]
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
            llm_model.save_adapter_weight(args.dir, f"{step_cnt}")

    llm_model.save_adapter_weight(args.dir)


def inference_callback(cur_pos, outputs):
    print(f"POSITION: {cur_pos}")
    for adapter_name, output in outputs.items():
        print(f"{adapter_name} OUTPUT: {output[0].strip()}")


def inference(adapters: Dict[str, mlora.LoraConfig],
              llm_model: mlora.LLMModel,
              tokenizer: mlora.Tokenizer,
              echo=False):
    gen_configs: List[mlora.GenerateConfig] = []
    for _, lora_config in adapters.items():
        gen_configs.append(mlora.GenerateConfig().init(lora_config))

    while True:
        input_raw = input("INPUT WITHOUT PROMPT: ")
        if input_raw == "QUIT":
            return
        for adapters in gen_configs:
            adapters.prompts_ = [input_raw]
        outputs = mlora.generate(llm_model, tokenizer, gen_configs,
                                 temperature=0.2,
                                 device=args.device,
                                 stream_callback=inference_callback if echo else None)
        print(f"\n{'='*10}\n")
        print(f"PROMPT: {input_raw}")
        for adapter_name, output in outputs.items():
            print(f"{adapter_name} OUTPUT:")
            print(output[0].strip())
        print(f"\n{'='*10}\n")


# Main Function
if __name__ == "__main__":
    setup_seed(args.seed)
    torch.set_default_device(args.device)

    with open(args.config, 'r', encoding='utf8') as fp:
        config = json.load(fp)

    tokenizer, model = load_base_model(config)
    adapters = init_lora_model(config, model)

    torch.cuda.empty_cache()

    if args.inference:
        inference(adapters, model, tokenizer, True)
    else:
        dispatcher = mlora.Dispatcher(config, tokenizer)
        train(adapters, config, model, dispatcher)
