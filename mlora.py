# m-LoRA: Efficient LLM Model Fine-Tune via Multi-LoRA Optimization
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
# Copyright (C) 2023-2024 All Rights Reserved.
#
# Github:  https://github.com/scukdde-llm/mlora

import os
import sys
import json
import torch
import mlora
import logging
import argparse
from typing import Dict, Tuple, List, Union

# Command Line Arguments
parser = argparse.ArgumentParser(description='m-LoRA main program')
parser.add_argument('--base_model', type=str, required=True,
                    help='Path to or name of base model')
parser.add_argument('--inference', action="store_true",
                    help='The inference mode (just for test)')
parser.add_argument('--evaluate', action="store_true",
                    help='The evaluate mode (just for test)')
parser.add_argument('--disable_prompter', action="store_true",
                    help='Disable prompter when inference')
parser.add_argument('--load_adapter', action="store_true",
                    help='Load adapter from file instead of init randomly')
parser.add_argument('--disable_adapter', action="store_true",
                    help="Disable the adapter modules")
parser.add_argument('--attn_impl', type=str,
                    help='Specify the implementation of attention')
parser.add_argument('--use_swa', action='store_true',
                    help='Use sliding window attention (requires flash attention)')
parser.add_argument('--fp16', action='store_true',
                    help='Load base model in float16 precision')
parser.add_argument('--bf16', action='store_true',
                    help='Load base model in bfloat16 precision')
parser.add_argument('--tf32', action="store_true",
                    help='Use tfloat32 instead of float32 if available')
parser.add_argument('--load_8bit', action="store_true",
                    help='Load base model with 8bit quantization')
parser.add_argument('--load_4bit', action="store_true",
                    help='Load base model with 4bit quantization')
parser.add_argument('--device', type=str,
                    help='Specify which GPU to be used')
parser.add_argument('--config', type=str, required=True,
                    help='Path to finetune configuration')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed in integer, default is 42')
parser.add_argument('--dir', type=str, default=".",
                    help='Path to read or save checkpoints')
parser.add_argument('--disable_log', action="store_true",
                    help='Disable logging.')
parser.add_argument('--log_file', type=str,
                    help='Save log to specific file')
parser.add_argument('--verbose', action="store_true",
                    help='Show extra informations such as parameters')
parser.add_argument('--overwrite', action="store_true",
                    help='Overwrite adapter model when older one existed')
parser.add_argument('--debug', action="store_true",
                    help='Enabling debugging mode')
parser.add_argument('--deterministic', action="store_true",
                    help='Use deterministic algorithms to improve the reproducibility')

args = parser.parse_args()


def query_yes_no(question, default="no"):
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write(
                "Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def load_base_model() -> Tuple[mlora.Tokenizer, mlora.LLMModel]:
    logging.info("Initializing pre-trained model.")
    model = mlora.LLMModel.from_pretrained(
        path=args.base_model,
        device=args.device,
        attn_impl=args.attn_impl,
        use_sliding_window=args.use_swa,
        bits=(8 if args.load_8bit else (4 if args.load_4bit else None)),
        load_dtype=(torch.bfloat16 if args.bf16 else (
            torch.float16 if args.fp16 else torch.float32))
    )

    tokenizer = mlora.Tokenizer(args.base_model)

    return tokenizer, model


def init_adapter_config(config: Dict[str, any],
                        llm_model: mlora.LLMModel,
                        ) -> List[Union[mlora.GenerateConfig, mlora.TrainConfig]]:
    config_list = []

    if config["cutoff_len"] == -1:
        config["cutoff_len"] = llm_model.max_seq_len_
        logging.info(
            f"Setting cutoff_len to {llm_model.max_seq_len_} automatically.")

    for lora_config in config["lora"]:
        lora_weight = None
        config_class = mlora.lora_config_factory(lora_config)
        config_class.adapter_name = lora_config["name"]
        config_class.task_name = lora_config.get("task_name", "casual")
        config_class.device = args.device

        adapter_file_path = args.dir + os.sep + \
            config_class.adapter_name + os.sep + "adapter_model.bin"
        if args.load_adapter:
            adapter_config_path = args.dir + os.sep + \
                config_class.adapter_name + os.sep + "adapter_config.json"
            logging.info(f"Load adapter: {adapter_file_path}")
            with open(adapter_config_path, 'r', encoding='utf8') as fp:
                adapter_config = json.load(fp)
                base_model_name_or_path = adapter_config.get(
                    "base_model_name_or_path", "")
                if base_model_name_or_path != "" and base_model_name_or_path != llm_model.name_or_path_:
                    raise ValueError("loading adapter with unmatched base model." +
                                     f" current is {llm_model.name_or_path_}, provided {base_model_name_or_path}")
            lora_weight = torch.load(
                adapter_file_path, map_location=args.device)
        elif os.path.isfile(adapter_file_path):
            if args.overwrite:
                logging.warning(
                    f"Overwriting existed adapter model file: {adapter_file_path}")
            elif not query_yes_no(f"Existed adapter model file detected: {adapter_file_path}\n" + "Overwrite?"):
                logging.info("User canceled training due to file conflict.")
                exit(0)

        if args.verbose:
            logging.info(config_class.__dict__)

        llm_model.init_lora_layer_weight(config_class, lora_weight)
        if args.inference:
            config_class = mlora.GenerateConfig(
                adapter_name=config_class.adapter_name)
            if not args.disable_prompter:
                config_class.prompt_template = lora_config.get("prompt", None)
            config_list.append(config_class)
        elif args.evaluate:
            if ';' in config_class.task_name:
                for task_name in config_class.task_name.split(';'):
                    config_list.append(mlora.EvaluateConfig(
                        adapter_name=config_class.adapter_name,
                        task_name=task_name,
                        batch_size=lora_config["test_batch_size"]))
            else:
                config_list.append(mlora.EvaluateConfig(
                    adapter_name=config_class.adapter_name,
                    task_name=config_class.task_name,
                    batch_size=lora_config["test_batch_size"]))
        else:
            config_list.append(mlora.TrainConfig(lora_config, config_class))

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
            config.prompts = [input_raw]
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
    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    if args.inference or args.evaluate:
        args.load_adapter = True

    is_train = not args.inference and not args.evaluate

    if args.attn_impl is None:
        if mlora.common._flash_attn_available:
            args.attn_impl = "flash_attn"
        else:
            args.attn_impl = "eager"

    mlora.setup_logging("INFO", args.log_file)

    mlora_backend = mlora.get_backend()

    if not mlora_backend.check_available():
        exit(-1)

    if args.device is None:
        args.device = f"{mlora_backend.device_name()}:0"

    mlora_backend.use_deterministic_algorithms(args.deterministic)
    mlora_backend.allow_tf32(args.tf32)
    mlora_backend.manual_seed(args.seed)

    with open(args.config, 'r', encoding='utf8') as fp:
        config = json.load(fp)

    tokenizer, model = load_base_model()
    adapters = init_adapter_config(config, model)

    mlora_backend.empty_cache()

    if args.inference:
        inference(model, tokenizer, adapters)
    elif args.evaluate:
        mlora.evaluate(model=model, tokenizer=tokenizer, configs=adapters,
                       max_concurrent_jobs=config.get(
                           "eval_lora_simultaneously_num", None),
                       retrying_steps=config.get(
                           "eval_rollback_retrying_steps", 20),
                       max_seq_len=config["cutoff_len"],
                       save_file=config.get("evaluate_result", None))
    else:
        mlora.train(mlora.Dispatcher(config, tokenizer), model,
                    adapters, args.dir, config["save_step"])
