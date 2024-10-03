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
# Copyright (C) 2024 All Rights Reserved.
#
# Github:  https://github.com/TUDB-Labs/mLoRA

import mlora.model
import mlora.utils
import mlora.executor
import mlora.config
import logging
import torch
from torch.utils.data import DataLoader
from dataset_utils import CustomDataset, evaluate_model  # Import from new file
import yaml

# Function to create the DataLoader using the dataset
def get_test_data_loader(config):
    # Creating the dataset using the provided configuration
    dataset = CustomDataset(config['data_path_'])

    # Creating the DataLoader from the dataset, using a configurable batch size
    batch_size = config.get('batch_size', 32)  # Default to 32 if not provided in config
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return data_loader

# Function to load YAML configuration file
def load_yaml_config(yaml_path):
    try:
        with open(yaml_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError as e:
        logging.error(f"Configuration file not found: {e}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error occurred while loading config: {e}")
        raise

if __name__ == "__main__":
    args = mlora.utils.get_cmd_args()

    #logging, seed, CUDA, and metric logger
    mlora.utils.setup_seed(args.seed)
    mlora.utils.setup_logging(args.log_level, args.log_file)
    mlora.utils.setup_cuda_check()
    mlora.utils.setup_metric_logger(args.metric_file)

    # Determining the device (CUDA if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Enable trace mode for profiling performance
    if args.trace:
        logging.info("Trace mode enabled for profiling performance.")
        mlora.utils.setup_trace_mode()

    # Load model and tokenizer
    logging.info("Loading model and tokenizer...")
    tokenizer, model = mlora.model.load_model(args)
    model = model.to(device)  # Move model to GPU or CPU as determined
    logging.info("Model and tokenizer loaded successfully.")

    # Load configuration from the YAML file
    logging.info(f"Loading configuration from {args.config}...")
    try:
        config = load_yaml_config(args.config)  # Use dynamic config path
    except Exception as e:
        logging.error(f"Failed to load config: {e}")
        exit(1)  # Exit if config loading fails

    logging.info("Configuration loaded.")

    # Initialize and add tasks
    logging.info("Initializing tasks...")
    executor = mlora.executor.Executor(model, tokenizer, config)
    for item in config['tasks_']:
        logging.info(f"Adding task: {item}")
        executor.add_task(item)

    # Execute tasks
    logging.info("Executing tasks...")
    executor.execute()
    logging.info("Task execution completed.")

    # Get test DataLoader using the dataset configuration
    test_data_loader = get_test_data_loader(config)

    # Evaluate the model
    accuracy = evaluate_model(model, test_data_loader, device)
    logging.info(f"Final model accuracy: {accuracy:.2f}%")