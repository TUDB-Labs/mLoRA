import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import datasets as hf_datasets
import evaluate as hf_evaluate
import torch

from mlora.common import InputData, Prompt


class BasicMetric:
    def __init__(self) -> None:
        pass

    def add_batch(self, predictions: torch.Tensor, references: torch.Tensor):
        pass

    def compute(self) -> Dict[str, Any]:
        pass


class AutoMetric(BasicMetric):
    def __init__(self, task_name: str) -> None:
        super().__init__()
        path_prefix = os.getenv("MLORA_METRIC_PATH")
        if path_prefix is None:
            path_prefix = ""
        elif not path_prefix.endswith(os.sep):
            path_prefix += os.sep

        if ":" in task_name:
            split = task_name.split(":")
            self.metric_ = hf_evaluate.load(path_prefix + split[0], split[1])
        else:
            self.metric_ = hf_evaluate.load(path_prefix + task_name)

    def add_batch(self, predictions: torch.Tensor, references: torch.Tensor):
        self.metric_.add_batch(predictions=predictions, references=references)

    def compute(self) -> Dict[str, Any]:
        return self.metric_.compute()


class BasicTask:
    def __init__(self) -> None:
        pass

    @property
    def peft_task_type(self) -> str:
        pass

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        pass

    def loading_metric(self) -> BasicMetric:
        pass

    def init_kwargs(self) -> Dict:
        return {}


# Casual Fine-tuning Tasks
# Instant-Created Class
class CasualTask(BasicTask):
    @property
    def peft_task_type(self) -> str:
        return "CAUSAL_LM"

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        assert path is not None, "Casual supervised fine-tuning requires data path."
        assert is_train, "Casual supervised fine-tuning task only supports training."
        # Loading dataset
        if path.endswith(".json") or path.endswith(".jsonl"):
            data = hf_datasets.load_dataset("json", data_files=path)
        elif ":" in path:
            split = path.split(":")
            data = hf_datasets.load_dataset(split[0], split[1])
        else:
            data = hf_datasets.load_dataset(path)
        ret: List[InputData] = []
        for data_point in data["train"]:
            ret.append(
                InputData(
                    inputs=Prompt(
                        instruction=data_point["instruction"],
                        input=data_point.get("input", None),
                        label=data_point.get("output", None),
                    )
                )
            )

        return ret


# Sequence Classification
class SequenceClassificationTask(BasicTask):
    def __init__(
        self,
        task_name: str,
        task_type: str,
        label_dtype: torch.dtype,
        num_labels: int,
        dataload_function: Callable,
        # Setting to `None` corresponds to the task name.
        metric_name: Optional[str] = None,
        # The default values are "train" and "validation".
        subset_map: Optional[Tuple[str, str]] = ("train", "validation"),
    ) -> None:
        super().__init__()
        self.task_name_ = task_name
        self.task_type_ = task_type
        self.label_dtype_ = label_dtype
        self.num_labels_ = num_labels
        self.dataload_function_ = dataload_function
        if metric_name is None:
            self.metric_name_ = task_name
        else:
            self.metric_name_ = metric_name
        self.subset_map_ = subset_map

    @property
    def peft_task_type(self) -> str:
        return "SEQ_CLS"

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        if ":" in self.task_name_:
            split = self.task_name_.split(":")
            data = hf_datasets.load_dataset(
                split[0] if path is None else path, split[1]
            )
        else:
            data = hf_datasets.load_dataset(self.task_name_ if path is None else path)
        data = data[self.subset_map_[0] if is_train else self.subset_map_[1]]
        logging.info(f"Preparing data for {self.task_name_.upper()}")
        ret: List[InputData] = []
        for data_point in data:
            inputs, labels = self.dataload_function_(data_point)
            assert isinstance(labels, List)
            ret.append(InputData(inputs=inputs, labels=labels))

        return ret

    def loading_metric(self) -> BasicMetric:
        return AutoMetric(self.metric_name_)

    def init_kwargs(self) -> Dict:
        return {
            "task_type": self.task_type_,
            "num_labels": self.num_labels_,
            "label_dtype": self.label_dtype_,
        }


# Common Sense
class CommonSenseTask(BasicTask):
    def __init__(self) -> None:
        super().__init__()
        self.task_type_ = "common_sense"
        self.label_dtype_ = None

    @property
    def peft_task_type(self) -> str:
        return "QUESTION_ANS"

    def label_list(self) -> List[str]:
        pass


task_dict = {}


# Multi-Task (Only for train)
class MultiTask(BasicTask):
    def __init__(self, task_names: str) -> None:
        super().__init__()
        self.task_type_ = "multi_task"
        self.label_dtype_ = None
        self.task_list_: List[BasicTask] = []
        task_names = task_names.split(";")
        for name in task_names:
            self.task_list_.append(task_dict[name])

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        logging.info(f"Preparing data for {len(self.task_list_)} tasks")
        path_list = None if path is None else path.split(";")
        data: List[InputData] = []
        assert is_train
        for idx, task in enumerate(self.task_list_):
            path: str = "" if path_list is None else path_list[idx].strip()
            data.extend(task.loading_data(is_train, None if len(path) == 0 else path))
        return data
