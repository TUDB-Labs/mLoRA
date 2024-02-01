from mlora.modelargs import LoraBatchDataConfig, MultiLoraBatchData
from mlora.tokenizer import Tokenizer
from mlora.prompter import Prompter
from mlora.model import LLMModel

from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass
import datasets as hf_datasets
import evaluate as hf_evaluate
import logging
import torch


class BasicTask():
    def __init__(self) -> None:
        pass

    def dataload_function(self, data_point) -> Tuple:
        return None, None, {"bos": True, "eos": True}

    def init_kwargs(self) -> Dict:
        return {}


# Casual Fine-tuning Tasks
class CasualTask(BasicTask):
    def __init__(self, prompter: Prompter = None) -> None:
        super().__init__()
        self.prompter_ = prompter

    def dataload_function(self, data_point) -> Tuple:
        return (self.prompter_.generate_prompt(
            data_point["instruction"],
            data_point.get("input", None),
            data_point.get("output", None)),
            None, {"bos": True, "eos": True})


# Sequence Classification
class SequenceClassification(BasicTask):
    def __init__(self,
                 task_type: str,
                 label_dtype: torch.dtype,
                 num_labels: int,
                 dataload_function: Callable) -> None:
        super().__init__()
        self.dataload_function_ = dataload_function
        self.label_dtype_ = label_dtype
        self.num_labels_ = num_labels
        self.task_type_ = task_type

    def dataload_function(self, data_point) -> Tuple:
        return self.dataload_function_(data_point)

    def init_kwargs(self) -> Dict:
        return {
            "task_type": self.task_type_,
            "num_labels": self.num_labels_,
            "label_dtype": self.label_dtype_,
        }


classification_tasks = {
    "glue:cola": SequenceClassification(
        task_type="single_label_classification",
        num_labels=2,
        label_dtype=torch.long,
        dataload_function=lambda data_point: (
            [data_point["sentence"]],
            [int(data_point["label"])],
            {"bos": True, "eos": True}
        ),
    ),
    "glue:mnli": SequenceClassification(
        task_type="single_label_classification",
        num_labels=3,
        label_dtype=torch.long,
        dataload_function=lambda data_point: (
            [data_point["premise"], data_point["hypothesis"]],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    ),
    "glue:mrpc": SequenceClassification(
        task_type="single_label_classification",
        num_labels=2,
        label_dtype=torch.long,
        dataload_function=lambda data_point: (
            [data_point["sentence1"], data_point["sentence2"]],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    ),
    "glue:qnli": SequenceClassification(
        task_type="single_label_classification",
        num_labels=2,
        label_dtype=torch.long,
        dataload_function=lambda data_point: (
            [data_point["question"], data_point["sentence"]],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    ),
    "glue:qqp": SequenceClassification(
        task_type="single_label_classification",
        num_labels=2,
        label_dtype=torch.long,
        dataload_function=lambda data_point: (
            [data_point["question1"], data_point["question2"]],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    ),
    "glue:rte": SequenceClassification(
        task_type="single_label_classification",
        num_labels=2,
        label_dtype=torch.long,
        dataload_function=lambda data_point: (
            [data_point["sentence1"], data_point["sentence2"]],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    ),
    "glue:sst2": SequenceClassification(
        task_type="single_label_classification",
        num_labels=2,
        label_dtype=torch.long,
        dataload_function=lambda data_point: (
            [data_point["sentence"]],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    ),
    "glue:wnli": SequenceClassification(
        task_type="single_label_classification",
        num_labels=2,
        label_dtype=torch.long,
        dataload_function=lambda data_point: (
            [data_point["sentence1"] + " </s> " + data_point["sentence2"]],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    ),
}


@dataclass
class EvaluateConfig:
    adapter_name_: str = None
    task_type_: str = None
    batch_size_: int = 16
    # Do not set these manually
    task_: SequenceClassification = None
    data_: hf_datasets.Dataset = None
    metric_: hf_evaluate.EvaluationModule = None
    batch_start_idx_: int = 0
    batch_end_idx_: int = 0

    def init_task(self):
        self.task_ = classification_tasks[self.task_type_]
        if ':' in self.task_type_:
            result = self.task_type_.split(':')
            self.data_ = hf_datasets.load_dataset(
                result[0], result[1])["validation"]
            self.metric_ = hf_evaluate.load(result[0], result[1])
        else:
            self.data_ = hf_datasets.load_dataset(
                self.task_type_)["validation"]
            self.metric_ = hf_evaluate.load(self.task_type_)

    def dataload(self, data_point):
        return self.task_.dataload_function_(data_point)


def _dispatch_task_in(tokenizer: Tokenizer, configs: List[EvaluateConfig], max_seq_len: int):
    batch_data_config = []
    current_configs = []
    batch_tokens = []
    batch_labels = []
    atten_masks = []
    for config in configs:
        if config.batch_start_idx_ >= len(config.data_):
            continue
        config.batch_end_idx_ = min(
            config.batch_start_idx_ + config.batch_size_, len(config.data_))
        batch_start_idx = len(batch_tokens)
        for idx in range(config.batch_start_idx_, config.batch_end_idx_):
            if idx >= len(config.data_):
                break
            texts, labels, kwargs = config.dataload(config.data_[idx])
            if "eos" in kwargs:
                kwargs["eos"] = False
            tokens = tokenizer.encode(texts, **kwargs)
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
            while len(tokens) < max_seq_len:
                tokens.append(tokenizer.pad_id_)
            batch_tokens.append(tokens)
            atten_masks.append(tokenizer.attention_mask(tokens))
            batch_labels.append(labels.copy())

        config.batch_start_idx_ = config.batch_end_idx_
        current_configs.append(config)
        batch_data_config.append(LoraBatchDataConfig(adapter_name_=config.adapter_name_,
                                                     batch_start_idx_=batch_start_idx, batch_end_idx_=len(batch_tokens)))

    return (current_configs,
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
             max_seq_len: int = 512):
    max_iterations = 0
    for config in configs:
        config.init_task()
        if len(config.data_) > max_iterations:
            max_iterations = len(config.data_)

    while True:
        current_configs, batch_labels, input_args = _dispatch_task_in(
            tokenizer, configs, max_seq_len)

        if len(current_configs) == 0:
            break

        outputs = model.forward(input_args)

        input_ids = torch.tensor(input_args.batch_tokens_, dtype=torch.long)

        for idx, output in enumerate(outputs):
            config: EvaluateConfig = current_configs[idx]
            task: SequenceClassification = config.task_
            metric = config.metric_
            start_idx = output.batch_start_idx_
            end_idx = output.batch_end_idx_
            logits = output.logits

            batch_size = logits.shape[0]
            sequence_lengths = (torch.eq(input_ids[start_idx:end_idx],
                                         tokenizer.pad_id_).int().argmax(-1) - 1).to(logits.device)
            pooled_logits = logits[torch.arange(batch_size,
                                                device=logits.device), sequence_lengths]
            labels = torch.tensor(batch_labels[start_idx:end_idx],
                                  dtype=task.label_dtype_, device=logits.device)
            if task.task_type_ == "single_label_classification":
                pooled_logits = torch.argmax(
                    pooled_logits, dim=-1).to(task.label_dtype_)
            elif task.task_type_ != "multi_label_classification":
                raise ValueError(f"unknown task type {task.task_type_}")

            metric.add_batch(predictions=pooled_logits.detach().cpu(),
                             references=labels.detach().cpu())
            logging.info(f"{config.adapter_name_} evaluate data:")
            logging.info(
                f"    step: {config.batch_start_idx_}/{len(config.data_)}")

    for config in configs:
        logging.info(f"{config.adapter_name_} evaluate result:")
        result = config.metric_.compute()
        for name, value in result.items():
            logging.info(f"    {name} = {value}")
