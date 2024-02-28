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

import mlora
import argparse
import logging

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
# the argument about pipeline
parser.add_argument('--pipeline', action="store_true",
                    help="Train the LoRA model use the pipeline parallelism")
parser.add_argument('--rank', type=int, default=-1,
                    help="The device's rank number")
parser.add_argument('--balance', type=int, nargs="+",
                    help="The model's balance")


args = parser.parse_args()


# to get test result and want early stop it
def train(config: mlora.MLoRAConfig, llm_model: mlora.LLMModel, dispatcher: mlora.Dispatcher):
    trainer = mlora.Trainer(llm_model, dispatcher, config.lora_configs_)
    trainer.train()


# Main Function
if __name__ == "__main__":
    # set the random seed
    mlora.setup_seed(args.seed)
    mlora.setup_logging(args.log_level, args.log_file)
    mlora.setup_cuda_check()

    # load part of model to device
    partial_model_to_device = None
    if args.pipeline:
        assert args.rank != -1
        assert len(args.balance) >= args.rank
        logging.info(
            f"Pipeline parallelism, rank is {args.rank} and balance is {args.balance}.")

        partial_model_to_device = [
            index + sum(args.balance[:args.rank])for index in range(0, args.balance[args.rank])]

    tokenizer, model = mlora.load_base_model(args.base_model,
                                             args.model_type,
                                             args.device,
                                             args.load_4bit,
                                             args.load_8bit,
                                             partial_model_to_device)

    if not args.disable_lora:
        assert args.config is not None, "error: Argument --config are required."
        config = mlora.MLoRAConfig(args.config)
        mlora.init_lora_model(model, config.lora_configs_)

    dispatcher = mlora.Dispatcher(config, tokenizer)

    if args.pipeline:
        pipe = mlora.Pipe(model,
                          config,
                          dispatcher,
                          args.device,
                          args.rank,
                          args.balance)
        exit(pipe.run())

    if args.evaluate:
        evaluator: mlora.Evaluator = mlora.EvaluatorFactory().create(
            model, tokenizer, args.evaluate, args.evaluate_data)
        evaluator.evaluate()
    else:
        train(config, model, dispatcher)
