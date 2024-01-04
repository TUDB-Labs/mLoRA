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
parser.add_argument('--load_adapter', action="store_true",
                    help="Load adapter from file instead of init randomly")
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
            bits=(8 if args.load_8bit else (4 if args.load_4bit else None)),
            verbose=args.log
        )
    elif args.model_type == "chatglm":
        log("Initializing ChatGLM model.")
        model = mlora.ChatGLMModel.from_pretrained(
            path=args.base_model,
            device=args.device,
            bits=(8 if args.load_8bit else (4 if args.load_4bit else None)),
            verbose=args.log
        )
    else:
        raise f"unkown model type {args.model_type}"

    tokenizer = mlora.Tokenizer(args.base_model, device=args.device)

    model.pad_token_id_ = tokenizer.pad_id_

    return tokenizer, model


def init_adapter_config(config: Dict[str, any], llm_model: mlora.LLMModel, do_train: bool = True) -> Dict[str, any]:
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

        if args.load_adapter:
            adapter_file_path = args.dir + os.sep + \
                config_class.adapter_name_ + os.sep + "adapter_model.bin"
            log(f"Load adapter: {adapter_file_path}")
            lora_weight = torch.load(
                adapter_file_path, map_location=args.device)

        llm_model.init_lora_layer_weight(config_class, lora_weight)
        if do_train:
            config_dict[config_class.adapter_name_] = mlora.TrainConfig().init(
                lora_config, config_class)
        else:
            config_dict[config_class.adapter_name_] = mlora.GenerateConfig().init(
                config_class)

    return config_dict


def inference_callback(cur_pos, outputs):
    print(f"POSITION: {cur_pos}")
    for adapter_name, output in outputs.items():
        print(f"{adapter_name} OUTPUT: {output[0].strip()}")


def inference(llm_model: mlora.LLMModel,
              tokenizer: mlora.Tokenizer,
              adapters: dict,
              verbose=False):
    gen_configs: List[mlora.GenerateConfig] = []
    for _, lora_config in adapters.items():
        lora_config.prompt_template_ = None
        gen_configs.append(mlora.GenerateConfig().init(lora_config))

    while True:
        input_raw = input("INPUT WITHOUT PROMPT: ")
        if input_raw == "QUIT":
            return
        for config in gen_configs:
            config.prompts_ = [input_raw]
        outputs = mlora.generate(llm_model, tokenizer, gen_configs,
                                 temperature=0.2,
                                 device=args.device,
                                 stream_callback=inference_callback if verbose else None)
        print(f"\n{'='*10}\n")
        print(f"PROMPT: {input_raw}")
        for adapter_name, output in outputs.items():
            print(f"{adapter_name} OUTPUT:")
            print(output[0].strip())
        print(f"\n{'='*10}\n")


# Main Function
if __name__ == "__main__":
    setup_seed(args.seed)

    with open(args.config, 'r', encoding='utf8') as fp:
        config = json.load(fp)

    tokenizer, model = load_base_model(config)
    adapters = init_adapter_config(config, model)

    torch.cuda.empty_cache()

    if args.inference:
        inference(model, tokenizer, adapters, args.log)
    else:
        mlora.train(mlora.Dispatcher(config, tokenizer), model,
                    adapters, args.device, args.dir, config["save_step"], args.log)
