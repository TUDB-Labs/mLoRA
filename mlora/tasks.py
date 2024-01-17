from mlora.prompter import Prompter
from mlora.model import LLMModel
from typing import Dict, Callable
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
             logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
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
             logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = torch.tensor(labels, dtype=torch.long, device=logits.device)
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(logits.contiguous().view(-1, self.vocab_size_), labels.contiguous().view(-1))


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

    def loss(self, input_ids: torch.Tensor,
             logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = torch.tensor(
            labels, dtype=self.label_dtype_, device=logits.device)
        batch_size = input_ids.shape[0]
        if self.pad_id_ is None:
            sequence_lengths = -1
        else:
            sequence_lengths = (
                torch.eq(input_ids, self.pad_id_).int().argmax(-1) - 1).to(logits.device)
        pooled_logits = logits[torch.arange(
            batch_size, device=logits.device), sequence_lengths]
        if self.task_type_ == "regression":
            loss_fn = torch.nn.MSELoss()
            return loss_fn(pooled_logits.squeeze(), labels.squeeze())
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
        "num_labels": 2,
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


def task_factory(model: LLMModel, train_config: Dict[str, any]):
    task_type = train_config.get("task_type", "casual")
    if task_type == "casual":
        return CasualLM(model, Prompter(train_config["prompt"]))
    else:
        return SequenceClassification(model, **classification_task_dict[task_type])
