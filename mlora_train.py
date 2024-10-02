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
from torch.utils.data import DataLoader, Dataset
import json
import yaml


# Custom dataset class to load data from JSON
class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        # Load the dataset from JSON
        with open(data_path, 'r') as f:
            data = json.load(f)
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item['input'], item['label']  # Adjust this according to your JSON format


# Evaluation function to evaluate the model
def evaluate_model(model, test_data_loader, device):
    logging.info("Starting evaluation...")
    model.eval()  # Setting model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_data_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to device
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    logging.info(f"Evaluation completed. Accuracy: {accuracy:.2f}%")
    return accuracy


# Function to create the DataLoader using the dataset
def get_test_data_loader(config):
    # Creating the dataset using the provided configuration
    dataset = CustomDataset(config['data_path_'])

    # Creating the DataLoader from the dataset
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    return data_loader


# Function to load YAML configuration file
def load_yaml_config(yaml_path):
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    args = mlora.utils.get_cmd_args()

    # Setup logging, seed, CUDA, and metric logger
    mlora.utils.setup_seed(args.seed)
    mlora.utils.setup_logging(args.log_level, args.log_file)
    mlora.utils.setup_cuda_check()
    mlora.utils.setup_metric_logger(args.metric_file)

    # Determine the device (CUDA if available)
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
    config = load_yaml_config('demo/vera/vera_case_1.yaml')
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