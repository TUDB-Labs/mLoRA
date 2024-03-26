from mlora.modelargs import DataClass
from mlora.tokenizer import Tokenizer
from mlora.prompter import Prompter

from typing import Any, Dict, List, Tuple, Optional, Callable
import datasets as hf_datasets
import evaluate as hf_evaluate
import logging
import torch


class BasicMetric():
    def __init__(self) -> None:
        pass

    def add_batch(self, predictions: torch.Tensor, references: torch.Tensor):
        pass

    def compute(self) -> Dict[str, Any]:
        pass


class AutoMetric(BasicMetric):
    def __init__(self, task_name: str) -> None:
        super().__init__()
        if ':' in task_name:
            split = task_name.split(':')
            self.metric_ = hf_evaluate.load(split[0], split[1])
        else:
            self.metric_ = hf_evaluate.load(task_name)

    def add_batch(self, predictions: torch.Tensor, references: torch.Tensor):
        self.metric_.add_batch(predictions=predictions, references=references)

    def compute(self) -> Dict[str, Any]:
        return self.metric_.compute()


class BasicTask():
    def __init__(self) -> None:
        pass

    def loading_data(self,
                     tokenizer: Tokenizer,
                     is_train: bool = True) -> List[DataClass]:
        pass

    def loading_metric(self) -> BasicMetric:
        pass

    def init_kwargs(self) -> Dict:
        return {}


# Casual Fine-tuning Tasks
# Instant-Created Class
class CasualTask(BasicTask):
    def __init__(self,
                 data_path: str,
                 prompt_template: str = None,
                 validation_size: int = None) -> None:
        super().__init__()
        # Loading dataset
        if data_path.endswith(".json") or data_path.endswith(".jsonl"):
            self.dataset_ = hf_datasets.load_dataset(
                "json", data_files=data_path)
        elif ':' in data_path:
            split = data_path.split(':')
            self.dataset_ = hf_datasets.load_dataset(split[0], split[1])
        else:
            self.dataset_ = hf_datasets.load_dataset(data_path)
        # Setup prompter
        self.prompter_ = Prompter(prompt_template)
        # Setup validation set
        if validation_size is not None:
            self.dataset_ = self.dataset_.train_test_split(
                test_size=validation_size)

    def loading_data(self,
                     tokenizer: Tokenizer,
                     is_train: bool = True) -> List[DataClass]:
        data = self.dataset_["train" if is_train else "test"]
        ret: List[DataClass] = []
        for idx, data_point in enumerate(data):
            prompt = self.prompter_.generate_prompt(
                data_point["instruction"],
                data_point.get("input", None),
                data_point.get("output", None))
            tokens = tokenizer.encode(data=prompt, bos=True, eos=False)
            ret.append(DataClass(tokens_=tokens, labels_=None))
            if idx % 10000 == 0:
                logging.info(f"Encode text data: {idx}/{len(data)}")

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

    def loading_data(self,
                     tokenizer: Tokenizer,
                     is_train: bool = True) -> List[DataClass]:
        if ':' in self.task_name_:
            split = self.task_name_.split(':')
            data = hf_datasets.load_dataset(split[0], split[1])
        else:
            data = hf_datasets.load_dataset(self.task_name_)
        data = data[self.subset_map_[0] if is_train else self.subset_map_[1]]
        logging.info(f"Preparing data for {self.task_name_.upper()}")
        ret: List[DataClass] = []
        for idx, data_point in enumerate(data):
            inputs, labels, flags = self.dataload_function_(data_point)
            assert isinstance(labels, List)
            if "eos" in flags and not is_train:
                flags["eos"] = False
            tokens = tokenizer.encode(data=inputs, **flags)
            ret.append(DataClass(tokens_=tokens, labels_=labels))
            if idx % 10000 == 0:
                logging.info(f"Encode text data: {idx}/{len(data)}")

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

    def label_list(self) -> List[str]:
        pass


task_dict = {}
