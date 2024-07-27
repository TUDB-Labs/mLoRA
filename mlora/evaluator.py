import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, List

import torch

from .common import InputData, LLMBatchConfig, LLMModelInput, MixConfig, Prompt
from .model import LLMModel
from .tasks import BasicMetric, BasicTask, CommonSenseTask, task_dict
from .tokenizer import Tokenizer


@dataclass
class EvaluateConfig:
    adapter_name: str = None
    task_name: str = None
    data_path: str = None
    batch_size: int = 16
    router_profile: bool = False
    # Do not set these manually
    task_: BasicTask = None
    data_: List[InputData] = None
    metric_: BasicMetric = None
    rollback_start_idx_: int = 0
    batch_start_idx_: int = 0
    batch_end_idx_: int = 0

    def _dataload_fn(self, tokenizer: Tokenizer, **tokenizer_kwargs):
        data = self.task_.loading_data(False, self.data_path)
        for idx, data_point in enumerate(data):
            assert not isinstance(data_point.inputs, Prompt)

            data_point.tokens = tokenizer.encode(data_point.inputs, **tokenizer_kwargs)
            if idx % 10000 == 0:
                logging.info(f"Encode text data: {idx}/{len(data)}")

        return data

    @staticmethod
    def from_config(config: Dict[str, any]) -> List["EvaluateConfig"]:
        adapter_name = config["name"]
        data_path = config.get("data", None)
        task_list = config.get("task_name", "casual").split(";")
        path_list = (
            [None] * len(task_list) if data_path is None else data_path.split(";")
        )
        config_list = []
        for task_name_, data_path_ in zip(task_list, path_list):
            if task_name_ not in task_dict:
                continue
            config_list.append(
                EvaluateConfig(
                    adapter_name=adapter_name,
                    task_name=task_name_,
                    data_path=data_path_,
                    batch_size=config["evaluate_batch_size"],
                )
            )

        return config_list

    def prepare(self, tokenizer: Tokenizer, device: str):
        self.reset_parameters()
        assert (
            self.task_name != "casual"
        ), "Auto evaluation is not currently available for casual supervised fine-tuning tasks."
        self.task_ = task_dict[self.task_name]
        self.data_ = self._dataload_fn(tokenizer)
        self.metric_ = self.task_.loading_metric()
        if isinstance(self.task_, CommonSenseTask):
            labels = self.task_.label_list()
            label_indices = [0] * len(labels)
            for idx, label in enumerate(labels):
                ids = tokenizer.encode(" " + label)
                label_indices[idx] = ids[-1]
            self.label_indices_ = torch.tensor(
                label_indices, dtype=torch.int64, device=device
            )
        else:
            self.label_indices_ = None

    def reset_parameters(self):
        self.task_ = None
        self.data_ = None
        self.metric_ = None
        self.rollback_start_idx_ = 0
        self.batch_start_idx_ = 0
        self.batch_end_idx_ = 0


def _prepare_tasks(model, tokenizer, configs):
    for config in configs:
        config.prepare(tokenizer, model.device_)
        if not isinstance(model.adapter_configs_[config.adapter_name], MixConfig):
            continue
        for layer in model.model_.layers_:
            if config.adapter_name in layer.mlp_.moes_:
                layer.mlp_.moes_[config.adapter_name].router_profile_ = (
                    config.router_profile
                )


def _dispatch_task_in(tokenizer, configs, concurrent_jobs, max_seq_len):
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
            config.batch_start_idx_ + config.batch_size, len(config.data_)
        )
        batch_start_idx = len(batch_tokens)
        for idx in range(config.batch_start_idx_, config.batch_end_idx_):
            if idx >= len(config.data_):
                break
            tokens = config.data_[idx].tokens
            labels = config.data_[idx].labels
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
            max_tokens_len = max(len(tokens), max_tokens_len)
            batch_tokens.append(tokens)
            batch_labels.append(labels.copy())

        config.batch_start_idx_ = config.batch_end_idx_
        current_configs.append(config)
        batch_data_config.append(
            LLMBatchConfig(
                adapter_name_=config.adapter_name,
                batch_start_idx_=batch_start_idx,
                batch_end_idx_=len(batch_tokens),
            )
        )

    max_seq_len = min(max_seq_len, max_tokens_len)

    for tokens in batch_tokens:
        sequence_lengths.append(len(tokens) - 1)
        while len(tokens) < max_seq_len:
            tokens.append(tokenizer.pad_id_)
        atten_masks.append(tokenizer.mask_from(tokens))

    return (
        current_configs,
        sequence_lengths,
        batch_labels,
        LLMModelInput(
            batch_configs_=batch_data_config,
            batch_tokens_=batch_tokens,
            batch_masks_=atten_masks,
            inference_mode_=True,
        ),
    )


def _compute_metrcis(model, current_configs, sequence_lengths, batch_labels, outputs):
    for idx, output in enumerate(outputs):
        config: EvaluateConfig = current_configs[idx]
        task: BasicTask = config.task_
        metric: BasicMetric = config.metric_
        start_idx = output.batch_start_idx_
        end_idx = output.batch_end_idx_
        logits = output.logits

        if config.router_profile:
            adapter_config = model.adapter_configs_[config.adapter_name]
            if isinstance(adapter_config, MixConfig):
                router_statistic_ = list(0 for _ in range(adapter_config.num_experts_))
                for layer in model.model_.layers_:
                    if config.adapter_name not in layer.mlp_.moes_:
                        continue
                    for idx, val in enumerate(
                        layer.mlp_.moes_[config.adapter_name].profiler_
                    ):
                        router_statistic_[idx] += val
                for idx, val in enumerate(router_statistic_):
                    logging.info(
                        f"{config.adapter_name}: expert {idx}, load = {val/32}"
                    )

        batch_size = logits.shape[0]
        pooled_logits = logits[
            torch.arange(batch_size, device=logits.device),
            sequence_lengths[start_idx:end_idx],
        ]
        labels = torch.tensor(
            batch_labels[start_idx:end_idx],
            dtype=task.label_dtype_,
            device=logits.device,
        )
        if task.task_type_ == "common_sense":
            pooled_logits = pooled_logits[:, config.label_indices_]
            pooled_logits = pooled_logits.softmax(-1).argmax(-1)
        elif task.task_type_ == "single_label_classification":
            pooled_logits = pooled_logits.softmax(-1).argmax(-1)
            pooled_logits = pooled_logits.to(task.label_dtype_)
        elif task.task_type_ != "multi_label_classification":
            raise ValueError(f"unknown task type {task.task_type_}")

        metric.add_batch(
            predictions=pooled_logits.detach().cpu(), references=labels.detach().cpu()
        )
        logging.info(f"{config.adapter_name} evaluate data:")
        logging.info(f"    step: {config.batch_start_idx_}/{len(config.data_)}")


def _compute_result(model, configs, save_file):
    results = []
    for config in configs:
        result = {
            "adapter_name": config.adapter_name,
            "task_name": config.task_name,
            "date_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "metrics": {},
        }
        compute_results = config.metric_.compute()
        result["metrics"] = compute_results
        if config.router_profile:
            adapter_config = model.adapter_configs_[config.adapter_name]
            if isinstance(adapter_config, MixConfig):
                router_statistic_ = list(0 for _ in range(adapter_config.num_experts_))
                for layer in model.model_.layers_:
                    if config.adapter_name not in layer.mlp_.moes_:
                        continue
                    for idx, val in enumerate(
                        layer.mlp_.moes_[config.adapter_name].profiler_
                    ):
                        router_statistic_[idx] += val
                    layer.mlp_.moes_[config.adapter_name].profiler_ = None
                result["router_profile"] = list(val / 32 for val in router_statistic_)

        results.append(result)

    if save_file is not None:
        with open(save_file, "w") as f:
            json.dump(results, f, indent=4)
        logging.info(f"saving evaluation result to {save_file}")
    else:
        print(json.dumps(results, indent=4))

    return results


@torch.inference_mode()
def evaluate(
    model: LLMModel,
    tokenizer: Tokenizer,
    configs: List[EvaluateConfig],
    max_concurrent_jobs: int = None,
    retrying_steps: int = 20,
    max_seq_len: int = 512,
    save_file: str = None,
) -> Dict:

    if max_concurrent_jobs is None:
        max_concurrent_jobs = len(configs)
        logging.info(
            f"Setting max_concurrent_jobs to {max_concurrent_jobs} automatically"
        )

    assert max_concurrent_jobs > 0
    assert retrying_steps > 0

    _prepare_tasks(model, tokenizer, configs)

    concurrent_jobs = max_concurrent_jobs
    retrying_count = 0
    while True:
        if concurrent_jobs < max_concurrent_jobs and retrying_count > 0:
            retrying_count -= 1
            if retrying_count == 0:
                concurrent_jobs += 1
                logging.info(f"recovering concurrent jobs to {concurrent_jobs}")

        current_configs, sequence_lengths, batch_labels, input_args = _dispatch_task_in(
            tokenizer, configs, concurrent_jobs, max_seq_len
        )

        if len(current_configs) == 0:
            break

        try:
            _compute_metrcis(
                model,
                current_configs,
                sequence_lengths,
                batch_labels,
                model.forward(input_args),
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                concurrent_jobs -= 1
                if concurrent_jobs == 0:
                    raise e
                logging.warn(
                    f"deprecating concurrent jobs to {concurrent_jobs} due to OOM."
                )
                # rollback
                retrying_count = retrying_steps
                for config in current_configs:
                    config.batch_start_idx_ = config.rollback_start_idx_
                    logging.info(
                        f"{config.adapter_name}: rollback to {config.batch_start_idx_}/{len(config.data_)}"
                    )
                continue
            else:
                raise e

        for config in current_configs:
            config.rollback_start_idx_ = config.batch_start_idx_

    return _compute_result(model, configs, save_file)
