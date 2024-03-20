from mlora.modelargs import DataClass, LoraBatchDataConfig, MultiLoraBatchData
from mlora.tasks.common import BasicTask, BasicMetric
from mlora.tasks.common import CommonSenseTask, task_dict
from mlora.tokenizer import Tokenizer
from mlora.model import LLMModel


from dataclasses import dataclass
from typing import List, Dict

import logging
import torch
import math
import json
import time


@dataclass
class EvaluateConfig:
    adapter_name_: str = None
    task_name_: str = None
    batch_size_: int = 16
    # Do not set these manually
    task_: BasicTask = None
    data_: List[DataClass] = None
    metric_: BasicMetric = None
    batch_start_idx_: int = 0
    batch_end_idx_: int = 0

    def prepare(self, tokenizer: Tokenizer, device: str = "cuda:0"):
        self.task_ = task_dict[self.task_name_]
        self.data_ = self.task_.loading_data(tokenizer, False)
        self.metric_ = self.task_.loading_metric()
        if isinstance(self.task_, CommonSenseTask):
            labels = self.task_.label_list()
            label_indices = [0] * len(labels)
            for idx, label in enumerate(labels):
                ids = tokenizer.encode(" " + label, False, False)
                label_indices[idx] = ids[-1]
            self.label_indices_ = torch.tensor(
                label_indices, dtype=torch.int64, device=device)
        else:
            self.label_indices_ = None


def _dispatch_task_in(tokenizer: Tokenizer, configs: List[EvaluateConfig], concurrent_jobs: int, max_seq_len: int):
    batch_data_config = []
    sequence_lengths = []
    current_configs = []
    batch_tokens = []
    batch_labels = []
    atten_masks = []
    max_tokens_len = 0
    for config in configs:
        if len(current_configs) >= concurrent_jobs:
            break
        if config.batch_start_idx_ >= len(config.data_):
            continue
        config.batch_end_idx_ = min(
            config.batch_start_idx_ + config.batch_size_, len(config.data_))
        batch_start_idx = len(batch_tokens)
        for idx in range(config.batch_start_idx_, config.batch_end_idx_):
            if idx >= len(config.data_):
                break
            tokens = config.data_[idx].tokens_
            labels = config.data_[idx].labels_
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
            max_tokens_len = max(len(tokens), max_tokens_len)
            # sequence_lengths.append(len(tokens))
            # while len(tokens) < max_seq_len:
            #     tokens.append(tokenizer.pad_id_)
            batch_tokens.append(tokens)
            # atten_masks.append(tokenizer.mask_from(tokens))
            batch_labels.append(labels.copy())

        config.batch_start_idx_ = config.batch_end_idx_
        current_configs.append(config)
        batch_data_config.append(LoraBatchDataConfig(adapter_name_=config.adapter_name_,
                                                     batch_start_idx_=batch_start_idx, batch_end_idx_=len(batch_tokens)))

    if max_tokens_len < max_seq_len:
        max_seq_len = math.ceil(max_tokens_len / 8) * 8

    for tokens in batch_tokens:
        sequence_lengths.append(len(tokens) - 1)
        while len(tokens) < max_seq_len:
            tokens.append(tokenizer.pad_id_)
        atten_masks.append(tokenizer.mask_from(tokens))

    return (current_configs,
            sequence_lengths,
            batch_labels,
            MultiLoraBatchData(
                lora_batch_data_config_=batch_data_config,
                batch_tokens_=batch_tokens,
                attention_masks_=atten_masks,
                gradient_checkpoint_=False))


@torch.inference_mode()
def evaluate(model: LLMModel,
             tokenizer: Tokenizer,
             configs: List[EvaluateConfig],
             concurrent_jobs: int = 2,
             max_seq_len: int = 512,
             save_file: str = None) -> Dict:

    max_iterations = 0
    for config in configs:
        config.prepare(tokenizer)
        if len(config.data_) > max_iterations:
            max_iterations = len(config.data_)

    while True:
        current_configs, sequence_lengths, batch_labels, input_args = _dispatch_task_in(
            tokenizer, configs, concurrent_jobs, max_seq_len)

        if len(current_configs) == 0:
            break

        outputs = model.forward(input_args)

        for idx, output in enumerate(outputs):
            config: EvaluateConfig = current_configs[idx]
            task: BasicTask = config.task_
            metric: BasicMetric = config.metric_
            start_idx = output.batch_start_idx_
            end_idx = output.batch_end_idx_
            logits = output.logits

            batch_size = logits.shape[0]
            pooled_logits = logits[torch.arange(
                batch_size, device=logits.device), sequence_lengths[start_idx:end_idx]]
            labels = torch.tensor(batch_labels[start_idx:end_idx],
                                  dtype=task.label_dtype_, device=logits.device)
            if task.task_type_ == "common_sense":
                pooled_logits = pooled_logits[:, config.label_indices_]
                pooled_logits = pooled_logits.softmax(-1).argmax(-1)
            elif task.task_type_ == "single_label_classification":
                pooled_logits = pooled_logits.softmax(-1).argmax(-1)
                pooled_logits = pooled_logits.to(task.label_dtype_)
            elif task.task_type_ != "multi_label_classification":
                raise ValueError(f"unknown task type {task.task_type_}")

            metric.add_batch(predictions=pooled_logits.detach().cpu(),
                             references=labels.detach().cpu())
            logging.info(f"{config.adapter_name_} evaluate data:")
            logging.info(
                f"    step: {config.batch_start_idx_}/{len(config.data_)}")

    results = []
    for config in configs:
        result = {
            "adapter_name": config.adapter_name_,
            "task_name": config.task_name_,
            "date_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "metrics": {}
        }
        logging.info(f"{config.adapter_name_} evaluate result:")
        compute_results = config.metric_.compute()
        result["metrics"] = compute_results
        results.append(result)
        for name, value in compute_results.items():
            logging.info(f"    {name} = {value}")

    if save_file is not None:
        with open(save_file, "w") as f:
            json.dump(results, f, indent=4)
        logging.info(f"saving evaluation result to {save_file}")

    return results
