from mlora.modelargs import LoraBatchDataConfig, MultiLoraBatchData
from mlora.tokenizer import Tokenizer
from mlora.prompter import Prompter
from mlora.model import LLMModel

from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass
import datasets as hf_datasets
import evaluate as hf_evaluate
import logging
import torch


class BasicTask():
    def __init__(self) -> None:
        pass

    def dataload_function(self, data_point):
        return None, None, {"bos": True, "eos": True}

    def load_state_dict(self, weight: Dict[str, torch.Tensor]):
        pass

    def state_dict(self):
        return {}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass

    def loss(self, input_ids: torch.Tensor,
             logits: torch.Tensor, labels: List) -> torch.Tensor:
        pass


# Casual Fine-tuning Tasks
class CasualLM(BasicTask):
    def __init__(self, model: LLMModel, prompter: Prompter = None) -> None:
        super().__init__()
        self.vocab_size_ = model.vocab_size_
        self.lm_head_ = model.output_
        self.prompter_ = prompter

    def dataload_function(self, data_point):
        return self.prompter_.generate_prompt(
            data_point["instruction"],
            data_point.get("input", None),
            data_point.get("output", None)), None, {"bos": True, "eos": True}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head_(hidden_states)

    def loss(self, input_ids: torch.Tensor,
             logits: torch.Tensor, labels: List) -> torch.Tensor:
        labels = torch.tensor(labels, dtype=torch.long, device=logits.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(logits[..., :-1, :].contiguous().view(-1, self.vocab_size_),
                       labels[..., 1:].contiguous().view(-1))


# Sequence Classification
class SequenceClassification(BasicTask):
    def __init__(self,
                 model: LLMModel,
                 task_type: str,
                 label_dtype: torch.dtype,
                 num_labels: int,
                 dataload_function: Callable) -> None:
        super().__init__()
        self.dataload_function_ = dataload_function
        self.label_dtype_ = label_dtype
        self.num_labels_ = num_labels
        self.task_type_ = task_type
        self.pad_id_ = model.pad_token_id_
        self.score_ = torch.nn.Linear(
            model.dim_, self.num_labels_, bias=False, dtype=torch.float32, device=model.device_)

    def dataload_function(self, data_point):
        return self.dataload_function_(data_point)

    def load_state_dict(self, weight: Dict[str, torch.Tensor]):
        with torch.no_grad():
            self.score_.weight.copy_(weight["classifier"])

    def state_dict(self):
        return {"classifier": self.score_.weight}

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.score_(hidden_states.to(torch.float32))

    def _pool_logits(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        if self.pad_id_ is None:
            sequence_lengths = -1
        else:
            sequence_lengths = (
                torch.eq(input_ids, self.pad_id_).int().argmax(-1) - 1).to(logits.device)
        return logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

    def evaluate(self, input_ids: torch.Tensor,
                 logits: torch.Tensor, labels: List) -> Tuple:
        labels = torch.tensor(
            labels, dtype=self.label_dtype_, device=logits.device)
        pooled_logits = self._pool_logits(input_ids, logits)
        if self.task_type_ == "regression":
            if self.num_labels_ == 1:
                pooled_logits = pooled_logits[:, 0]
        elif self.task_type_ == "single_label_classification":
            pooled_logits = torch.argmax(
                pooled_logits, dim=-1).to(self.label_dtype_)
        elif self.task_type_ != "multi_label_classification":
            raise ValueError(f"unknown task type {self.task_type_}")
        return pooled_logits, labels.squeeze()

    def loss(self, input_ids: torch.Tensor,
             logits: torch.Tensor, labels: List) -> torch.Tensor:
        labels = torch.tensor(
            labels, dtype=self.label_dtype_, device=logits.device)
        pooled_logits = self._pool_logits(input_ids, logits)
        if self.task_type_ == "regression":
            loss_fn = torch.nn.MSELoss()
            if self.num_labels_ == 1:
                return loss_fn(pooled_logits.squeeze(), labels.squeeze())
            else:
                return loss_fn(pooled_logits, labels)
        elif self.task_type_ == "single_label_classification":
            loss_fn = torch.nn.CrossEntropyLoss()
            return loss_fn(pooled_logits.view(-1, self.num_labels_), labels.view(-1))
        elif self.task_type_ == "multi_label_classification":
            loss_fn = torch.nn.BCEWithLogitsLoss()
            return loss_fn(pooled_logits, labels)
        else:
            raise ValueError(f"unknown task type {self.task_type_}")


classification_task_dict = {
    "glue:cola": {
        "task_type": "single_label_classification",
        "num_labels": 2,
        "label_dtype": torch.long,
        "dataload_function": lambda data_point: (
            data_point["sentence"],
            [int(data_point["label"])],
            {"bos": True, "eos": True}
        ),
    },
    "glue:mnli": {
        "task_type": "single_label_classification",
        "num_labels": 3,
        "label_dtype": torch.long,
        "dataload_function": lambda data_point: (
            data_point["premise"] + " </s> " + data_point["hypothesis"],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    },
    "glue:mrpc": {
        "task_type": "single_label_classification",
        "num_labels": 2,
        "label_dtype": torch.long,
        "dataload_function": lambda data_point: (
            data_point["sentence1"] + " </s> " + data_point["sentence2"],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    },
    "glue:qnli": {
        "task_type": "single_label_classification",
        "num_labels": 2,
        "label_dtype": torch.long,
        "dataload_function": lambda data_point: (
            data_point["question"] + " </s> " + data_point["sentence"],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    },
    "glue:qqp": {
        "task_type": "single_label_classification",
        "num_labels": 2,
        "label_dtype": torch.long,
        "dataload_function": lambda data_point: (
            data_point["question1"] + " </s> " + data_point["question2"],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    },
    "glue:rte": {
        "task_type": "single_label_classification",
        "num_labels": 2,
        "label_dtype": torch.long,
        "dataload_function": lambda data_point: (
            data_point["sentence1"] + " </s> " + data_point["sentence2"],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    },
    "glue:sst2": {
        "task_type": "single_label_classification",
        "num_labels": 2,
        "label_dtype": torch.long,
        "dataload_function": lambda data_point: (
            data_point["sentence"],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    },
    "glue:stsb": {
        "task_type": "regression",
        "num_labels": 1,
        "label_dtype": torch.float,
        "dataload_function": lambda data_point: (
            data_point["sentence1"] + " </s> " + data_point["sentence2"],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    },
    "glue:wnli": {
        "task_type": "single_label_classification",
        "num_labels": 2,
        "label_dtype": torch.long,
        "dataload_function": lambda data_point: (
            data_point["sentence1"] + " </s> " + data_point["sentence2"],
            [int(data_point["label"])],
            {"bos": True, "eos": True},
        ),
    },
}


def train_task_factory(model: LLMModel, train_config: Dict[str, any], weight: Dict[str, torch.Tensor]) -> BasicTask:
    task_type = train_config.get("task_type", "casual")
    if task_type == "casual":
        task = CasualLM(model, Prompter(train_config["prompt"]))
    else:
        task = SequenceClassification(
            model, **classification_task_dict[task_type])
    if weight is not None:
        task.load_state_dict(weight)
    return task


def classification_task_factory(model: LLMModel, task_type: str, weight: Dict[str, torch.Tensor]) -> SequenceClassification:
    task = SequenceClassification(model, **classification_task_dict[task_type])
    if weight is not None:
        task.load_state_dict(weight)
    return task


@dataclass
class EvaluateConfig:
    adapter_name_: str = None
    task_type_: str = None
    task_: SequenceClassification = None
    batch_size_: int = 16,
    batch_seq_len_: int = 512
    # Do not set these manually
    data_: hf_datasets.Dataset = None
    metric_: hf_evaluate.EvaluationModule = None
    batch_start_idx_: int = 0
    batch_end_idx_: int = 0

    def init_task(self):
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
        return self.task_.dataload_function(data_point)


@torch.inference_mode()
def evaluate(model: LLMModel,
             tokenizer: Tokenizer,
             configs: List[EvaluateConfig],
             max_seq_len: int = 512):
    device = torch.device(model.device_)
    max_iterations = 0
    for config in configs:
        config.init_task()
        if len(config.data_) > max_iterations:
            max_iterations = len(config.data_)

    while True:
        batch_data_config = []
        current_configs = []
        batch_tokens = []
        batch_labels = []
        tokens_len = []
        for config in configs:
            if config.batch_start_idx_ >= len(config.data_):
                continue
            config.batch_end_idx_ = min(
                config.batch_start_idx_ + config.batch_size_, len(config.data_))
            batch_start_idx = len(batch_tokens)
            for idx in range(config.batch_start_idx_, config.batch_end_idx_):
                if idx >= len(config.data_):
                    break
                text, label, kwargs = config.dataload(config.data_[idx])
                batch_labels.append(label)
                tokens = tokenizer.encode(data=text, **kwargs)
                if len(tokens) > max_seq_len:
                    tokens = tokens[:max_seq_len]
                tokens_len.append(len(tokens))
                while len(tokens) < max_seq_len:
                    tokens.append(tokenizer.pad_id_)
                batch_tokens.append(tokens)

            config.batch_start_idx_ = config.batch_end_idx_
            current_configs.append(config)
            batch_data_config.append(LoraBatchDataConfig(adapter_name_=config.adapter_name_,
                                     batch_start_idx_=batch_start_idx, batch_end_idx_=len(batch_tokens)))

        if len(current_configs) == 0:
            break

        batch_tokens = torch.tensor(
            batch_tokens, dtype=torch.long, device=device)

        input_data = MultiLoraBatchData(
            lora_batch_data_config_=batch_data_config,
            batch_seq_len_=max_seq_len,
            expand_side_=["right"]*batch_tokens.shape[0],
            batch_tokens_=batch_tokens,
            tokens_len_without_pad_=tokens_len,
            checkpoint_recompute_=False,
            inference_seq_pos_=-1)

        outputs = model.forward(input_data)[0]

        for idx, config in enumerate(batch_data_config):
            task_config = current_configs[idx]
            task = task_config.task_
            metric = task_config.metric_
            start_idx = config.batch_start_idx_
            end_idx = config.batch_end_idx_
            input_ids = batch_tokens[start_idx:end_idx]
            logits = task.forward(outputs[start_idx:end_idx])
            predictions, references = task.evaluate(
                input_ids, logits, batch_labels[start_idx:end_idx])
            metric.add_batch(predictions=predictions, references=references)
            logging.info(f"{config.adapter_name_} evaluate data:")
            logging.info(
                f"    step: {task_config.batch_start_idx_}/{len(task_config.data_)}")

    for config in configs:
        logging.info(f"{config.adapter_name_} evaluate result:")
        result = config.metric_.compute()
        for name, value in result.items():
            logging.info(f"    {name} = {value}")
