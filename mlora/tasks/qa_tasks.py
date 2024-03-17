from mlora.tasks.common import BasicMetric, AutoMetric
from mlora.tasks.common import CommonSenseTask
from mlora.modelargs import DataClass
from mlora.tokenizer import Tokenizer
from typing import List

import datasets as hf_datasets
import logging
import torch


class OpenBookQA(CommonSenseTask):
    def __init__(self) -> None:
        super().__init__()
        self.label_dtype_ = torch.int
        self.labels_ = ["A", "B", "C", "D"]
        self.labels2id_ = {text: idx for idx, text in enumerate(self.labels_)}

    def label_list(self) -> List[str]:
        return self.labels_

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

    def loading_metric(self) -> BasicMetric:
        return AutoMetric("accuracy")


def update_task_dict(task_dict):
    task_dict.update({
        "obqa": OpenBookQA()
    })
