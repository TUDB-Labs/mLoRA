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

import json
import torch
import mlora
import argparse
from typing import Dict, List

# Command Line Arguments
parser = argparse.ArgumentParser(description='m-LoRA main program')
parser.add_argument('--base_model', type=str, required=True,
                    help='Path to or name of base model')
parser.add_argument('--tokenizer', type=str,
                    help='Path to or name of tokenizer')
parser.add_argument('--model_type', type=str, default="llama",
                    help='The model type, support: llama, chatglm')
parser.add_argument('--device', type=str, default='cuda:0',
                    help='Specify which GPU to be used, default is cuda:0')
# load quant
parser.add_argument('--load_8bit', action="store_true",
                    help='Load model in 8bit mode')
parser.add_argument('--load_4bit', action="store_true",
                    help='Load model in 4bit mode')
# inference model
parser.add_argument('--inference', action="store_true",
                    help='The inference mode (just for test)')
# mmlu evaluate model
parser.add_argument('--evaluate', type=str,
                    help='Enable the evaluate mode.')
parser.add_argument('--evaluate_data', type=str,
                    help='The evaluate dataset name or path.')
# whether to enable the lora
parser.add_argument('--load_lora', action="store_true",
                    help="Load lora from file instead of init randomly")
parser.add_argument('--disable_lora', action="store_true",
                    help="Disable the lora modules")
# configuration
parser.add_argument('--config', type=str,
                    help='Path to finetune configuration')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed in integer, default is 42')
# configuration about log
parser.add_argument('--log_level', type=str, default="INFO",
                    help="Set the log level.")
parser.add_argument('--log_file', type=str,
                    help="Save log to specific file.")

args = parser.parse_args()


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


# to get test result and want early stop it
def train(config: Dict[str, any], llm_model: mlora.LLMModel, dispatcher: mlora.Dispatcher):
    # the train paramas per lora model
    all_train_paramas: Dict[str, List[torch.Tensor]
                            ] = llm_model.get_train_paramas()
    all_optimizer: Dict[str, torch.optim.Optimizer] = get_optimizer(
        config, all_train_paramas)
    accumulation_step: Dict[str, int] = get_accumulation_steps(config)

    loss_fn = torch.nn.CrossEntropyLoss()

    step_cnt = 0
    while not dispatcher.check_task_done():
        input: mlora.MultiLoraBatchData = dispatcher.get_train_data()

        step_cnt += 1

        output = llm_model.forward(input)
        labels = torch.tensor(input.batch_tokens_,
                              dtype=torch.long).to(args.device)

        total_loss = None
        for lora_config in input.lora_batch_data_config_:
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_
            loss_input = output[start_idx:end_idx][..., :-1,
                                                   :].contiguous().view(-1, llm_model.vocab_size_)
            loss_target = labels[start_idx:end_idx][...,
                                                    1:].contiguous().view(-1)
            loss = loss_fn(loss_input, loss_target) / \
                accumulation_step[lora_config.adapter_name_]
            print(
                f"    adapter: {lora_config.adapter_name_} loss: {loss}")
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        total_loss.backward()
        for lora in input.lora_batch_data_config_:
            if step_cnt % accumulation_step[lora.adapter_name_] == 0:
                all_optimizer[lora.adapter_name_].step()
                all_optimizer[lora.adapter_name_].zero_grad()

        if step_cnt % config["save_step"] == 0:
            mlora.save_lora_model(llm_model, config, f"{step_cnt}")

    mlora.save_lora_model(llm_model, config)


def inference(config: Dict[str, any],
              llm_model: mlora.LLMModel,
              tokenizer: mlora.Tokenizer):
    lora_adapter_num = len(config["lora"])
    batch_data_config: List[mlora.LoraBatchDataConfig] = []

    for idx, lora_config in enumerate(config["lora"]):
        adapter_name = lora_config["name"]
        batch_data_config.append(mlora.LoraBatchDataConfig(
            adapter_name, idx, idx + 1))

    inference_max_len = 128

    while True:
        input_raw = input("INPUT WITHOUT PROMPT: ")
        if input_raw == "QUIT":
            return

        tokens = tokenizer.encode(input_raw, True, False)
        token_len = len(tokens)
        while len(tokens) < inference_max_len:
            tokens.append(tokenizer.pad_id_)

        input_data = mlora.MultiLoraBatchData(
            prompts_=[input_raw] * lora_adapter_num,
            lora_batch_data_config_=batch_data_config,
            batch_tokens_=[tokens] * lora_adapter_num,
            tokens_len_without_pad_=[token_len] * lora_adapter_num,
            batch_seq_len_=inference_max_len,
            expand_side_=["right"] * lora_adapter_num,
            inference_model_=True)

        eos_flag: List[bool] = [False] * lora_adapter_num
        for pos in range(token_len, inference_max_len):
            with torch.no_grad():
                # batch_size, seq_len, voc_logs
                outputs = llm_model.forward(input_data)
                next_token = outputs[:, pos - 1, :]
                next_token = torch.argmax(next_token, dim=-1)
                for idx in range(len(input_data.batch_tokens_)):
                    input_data.batch_tokens_[idx][pos] = next_token[idx].item()
                    # end of the sentence
                    if next_token[idx].item() == tokenizer.eos_id_:
                        eos_flag[idx] = True
                    input_data.tokens_len_without_pad_[
                        idx] = input_data.tokens_len_without_pad_[idx] + 1
            # check if the all sentence end
            have_all_done = all(flag for flag in eos_flag)
            if have_all_done:
                break

        for idx, output in enumerate(input_data.batch_tokens_):
            print(f"# LORA{idx} OUTPUT IS:")
            print(tokenizer.decode(output))


# Main Function
if __name__ == "__main__":
    # set the random seed
    mlora.setup_seed(args.seed)
    mlora.setup_logging(args.log_level, args.log_file)
    mlora.setup_cuda_check()

    tokenizer, model = mlora.load_base_model(args.base_model,
                                             args.model_type,
                                             args.device,
                                             args.load_4bit,
                                             args.load_8bit)

    if not args.disable_lora:
        assert args.config is not None, "error: Argument --config are required."

        with open(args.config, 'r', encoding='utf8') as fp:
            config = json.load(fp)
        mlora.init_lora_model(config, model, args.load_lora)

    if args.inference:
        inference(config, model, tokenizer)
    elif args.evaluate:
        evaluator: mlora.Evaluator = mlora.EvaluatorFactory().create(
            model, tokenizer, args.evaluate, args.evaluate_data)
        evaluator.evaluate()
    else:
        dispatcher = mlora.Dispatcher(config, tokenizer)
        train(config, model, dispatcher)
