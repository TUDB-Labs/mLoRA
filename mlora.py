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
import logging
import argparse
from typing import Dict, Tuple, List, Union

# Command Line Arguments
parser = argparse.ArgumentParser(description='m-LoRA main program')
parser.add_argument('--base_model', type=str, required=True,
                    help='Path to or name of base model')
parser.add_argument('--model_type', type=str, default="llama",
                    help='The model type, support: llama, chatglm')
parser.add_argument('--inference', action="store_true",
                    help='The inference mode (just for test)')
parser.add_argument('--evaluate', action="store_true",
                    help='The evaluate mode (just for test)')
parser.add_argument('--disable_prompter', action="store_true",
                    help="Disable prompter when inference")
parser.add_argument('--load_adapter', action="store_true",
                    help="Load adapter from file instead of init randomly")
parser.add_argument('--disable_adapter', action="store_true",
                    help="Disable the adapter modules")
parser.add_argument('--tokenizer', type=str,
                    help='Path to or name of tokenizer')
parser.add_argument('--load_16bit', action='store_true',
                    help='Load model in half precision')
parser.add_argument('--load_8bit', action="store_true",
                    help='Load model in 8bit mode')
parser.add_argument('--load_4bit', action="store_true",
                    help='Load model in 4bit mode')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='Specify which GPU to be used, default is cuda:0')
parser.add_argument('--config', type=str, required=True,
                    help='Path to finetune configuration')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed in integer, default is 42')
parser.add_argument('--dir', type=str, default=".",
                    help='Path to read or save checkpoints')
parser.add_argument('--disable_log', action="store_true",
                    help='Disable logging.')
parser.add_argument('--log_file', type=str,
                    help="Save log to specific file.")

args = parser.parse_args()


# Functions
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def load_base_model() -> Tuple[mlora.Tokenizer, mlora.LLMModel]:
    if args.model_type == "llama":
        logging.info("Initializing LLaMA model.")
        model = mlora.LlamaModel.from_pretrained(
            path=args.base_model,
            device=args.device,
            bits=(8 if args.load_8bit else (4 if args.load_4bit else None)),
            load_dtype=torch.bfloat16 if args.load_16bit else torch.float32
        )
    elif args.model_type == "chatglm":
        logging.info("Initializing ChatGLM model.")
        model = mlora.ChatGLMModel.from_pretrained(
            path=args.base_model,
            device=args.device,
            bits=(8 if args.load_8bit else (4 if args.load_4bit else None)),
            load_dtype=torch.float32
        )
    else:
        raise f"unkown model type {args.model_type}"

    tokenizer = mlora.Tokenizer(args.base_model)

    return tokenizer, model


def init_adapter_config(config: Dict[str, any],
                        llm_model: mlora.LLMModel,
                        ) -> List[Union[mlora.GenerateConfig, mlora.TrainConfig]]:
    config_list = []

    for lora_config in config["lora"]:
        lora_weight = None
        config_class = mlora.lora_config_factory(lora_config)
        config_class.adapter_name_ = lora_config["name"]
        config_class.task_type_ = lora_config.get("task_type", "casual")
        config_class.device_ = args.device

        if args.load_adapter:
            adapter_file_path = args.dir + os.sep + \
                config_class.adapter_name_ + os.sep + "adapter_model.bin"
            logging.info(f"Load adapter: {adapter_file_path}")
            lora_weight = torch.load(
                adapter_file_path, map_location=args.device)

        llm_model.init_lora_layer_weight(config_class, lora_weight)
        if args.inference:
            config_class = mlora.GenerateConfig(
                adapter_name_=config_class.adapter_name_)
            if not args.disable_prompter:
                config_class.prompt_template_ = lora_config["prompt"]
        elif args.evaluate:
            config_class = mlora.EvaluateConfig(
                adapter_name_=config_class.adapter_name_,
                task_type_=config_class.task_type_,
                batch_size_=lora_config["micro_batch_size"])
        else:
            config_class = mlora.TrainConfig(lora_config, config_class)
        config_list.append(config_class)

    return config_list


def inference_callback(cur_pos, outputs):
    print(f"POSITION: {cur_pos}")
    for adapter_name, output in outputs.items():
        print(f"{adapter_name} OUTPUT: {output[0]}")


def inference(llm_model: mlora.LLMModel,
              tokenizer: mlora.Tokenizer,
              adapters: List[mlora.GenerateConfig]):
    while True:
        input_raw = input("INPUT WITHOUT PROMPT: ")
        if input_raw == "QUIT":
            return
        for config in adapters:
            config.prompts_ = [input_raw]
        callback = None if args.disable_log else inference_callback
        outputs = mlora.generate(llm_model, tokenizer, adapters,
                                 stream_callback=callback)
        print(f"\n{'='*10}\n")
        print(f"PROMPT: {input_raw}")
        for adapter_name, output in outputs.items():
            print(f"{adapter_name} OUTPUT:")
            print(output[0])
        print(f"\n{'='*10}\n")


# Main Function
if __name__ == "__main__":
    if args.inference or args.evaluate:
        args.load_adapter = True

    log_handlers = [logging.StreamHandler()]
    if args.log_file is not None:
        log_handlers.append(logging.FileHandler(args.log_file))

    logging.basicConfig(format='[%(asctime)s] m-LoRA: %(message)s',
                        level=logging.WARNING if args.disable_log else logging.INFO,
                        handlers=log_handlers,
                        force=True)

    if torch.cuda.is_available():
        logging.info('NVIDIA CUDA initialized successfully.')
        logging.info('Total %i GPU(s) detected.' % torch.cuda.device_count())
    else:
        logging.error(
            'm-LoRA requires NVIDIA CUDA computing capacity. Please check your PyTorch installation.')
        exit(-1)

    setup_seed(args.seed)

    with open(args.config, 'r', encoding='utf8') as fp:
        config = json.load(fp)

    tokenizer, model = load_base_model()
    adapters = init_adapter_config(config, model)

    torch.cuda.empty_cache()

    if args.inference:
        inference(model, tokenizer, adapters)
    elif args.evaluate:
        mlora.evaluate(model, tokenizer, adapters, config["cutoff_len"])
    else:
        mlora.train(mlora.Dispatcher(config, tokenizer), model,
                    adapters, args.dir, config["save_step"])
