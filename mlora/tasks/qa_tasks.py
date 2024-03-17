from mlora.tasks.common import BasicMetric, AutoMetric
from mlora.tasks.common import CommonSenseTask
from mlora.modelargs import DataClass
from mlora.tokenizer import Tokenizer
from typing import List

import datasets as hf_datasets
import logging
import torch


class QA(CommonSenseTask):
    def __init__(self, labels: List[str]) -> None:
        super().__init__()
        self.labels_ = labels
        self.labels2id_ = {text: idx for idx, text in enumerate(self.labels_)}
        self.label_dtype_ = torch.int

    def label_list(self) -> List[str]:
        return self.labels_

    def loading_metric(self) -> BasicMetric:
        return AutoMetric("accuracy")


class ARC(QA):
    def __init__(self, subject: str) -> None:
        super().__init__(["1", "2", "3", "4", "A", "B", "C", "D", "E"])
        assert subject in ["ARC-Easy", "ARC-Challenge"]
        self.subject_ = subject

    def loading_data(self,
                     tokenizer: Tokenizer,
                     is_train: bool = True) -> List[DataClass]:
        data = hf_datasets.load_dataset(
            "allenai/ai2_arc", self.subject_)["train" if is_train else "test"]
        ret: List[DataClass] = []
        for idx, data_point in enumerate(data):
            prompt = "Please choose the correct answer to the question: " + \
                data_point["question"]
            choices = data_point["choices"]
            for label, text in zip(choices["label"], choices["text"]):
                prompt += f" ({label}) {text}"
            prompt += "\nAnswer:"
            if is_train:
                prompt += " " + data_point["answerKey"]
                labels = None
            else:
                labels = [self.labels2id_[data_point["answerKey"]]]
            tokens = tokenizer.encode(data=prompt, bos=True, eos=False)
            ret.append(DataClass(tokens_=tokens, labels_=labels))
            if idx % 10000 == 0:
                logging.info(f"Encode text data: {idx}/{len(data)}")

        return ret


class OpenBookQA(QA):
    def __init__(self) -> None:
        super().__init__(["A", "B", "C", "D"])

    def loading_data(self,
                     tokenizer: Tokenizer,
                     is_train: bool = True) -> List[DataClass]:
        data = hf_datasets.load_dataset(
            "allenai/openbookqa", "main")["train" if is_train else "test"]
        ret: List[DataClass] = []
        for idx, data_point in enumerate(data):
            prompt = "Please choose the correct answer to the question: " + \
                data_point["question_stem"]
            choices = data_point["choices"]
            for label, text in zip(choices["label"], choices["text"]):
                prompt += f" ({label}) {text}"
            prompt += "\nAnswer:"
            if is_train:
                prompt += " " + data_point["answerKey"]
                labels = None
            else:
                labels = [self.labels2id_[data_point["answerKey"]]]
            tokens = tokenizer.encode(data=prompt, bos=True, eos=False)
            ret.append(DataClass(tokens_=tokens, labels_=labels))
            if idx % 10000 == 0:
                logging.info(f"Encode text data: {idx}/{len(data)}")

        return ret


def update_task_dict(task_dict):
    task_dict.update({
        "arc-e": ARC("ARC-Easy"),
        "arc-c": ARC("ARC-Challenge"),
        "obqa": OpenBookQA(),
    })
