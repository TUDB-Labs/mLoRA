import logging
from typing import List, Optional

import datasets as hf_datasets
import torch

from mlora.common import InputData

from .common import AutoMetric, BasicMetric, CommonSenseTask


class QuestionAnswerTask(CommonSenseTask):
    def __init__(self, labels: List[str]) -> None:
        super().__init__()
        self.labels_ = labels
        self.labels2id_ = {text: idx for idx, text in enumerate(self.labels_)}
        self.label_dtype_ = torch.int

    def label_list(self) -> List[str]:
        return self.labels_

    def loading_metric(self) -> BasicMetric:
        return AutoMetric("accuracy")


class ARC(QuestionAnswerTask):
    def __init__(self, subject: str) -> None:
        super().__init__(["1", "2", "3", "4", "A", "B", "C", "D", "E"])
        assert subject in ["ARC-Easy", "ARC-Challenge"]
        self.subject_ = subject

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        data = hf_datasets.load_dataset(
            "allenai/ai2_arc" if path is None else path, self.subject_
        )["train" if is_train else "test"]
        logging.info(f"Preparing data for {self.subject_}")
        ret: List[InputData] = []
        for data_point in data:
            prompt = (
                "Please choose the correct answer to the question: "
                + data_point["question"]
            )
            choices = data_point["choices"]
            for label, text in zip(choices["label"], choices["text"]):
                prompt += f" ({label}) {text}"
            prompt += "\nAnswer:"
            if is_train:
                prompt += " " + data_point["answerKey"]
                labels = None
            else:
                labels = [self.labels2id_[data_point["answerKey"]]]
            ret.append(InputData(inputs=prompt, labels=labels))

        return ret


class BoolQ(QuestionAnswerTask):
    def __init__(self) -> None:
        super().__init__(["true", "false"])

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        data = hf_datasets.load_dataset("google/boolq" if path is None else path)[
            "train" if is_train else "validation"
        ]
        logging.info("Preparing data for BoolQ")
        ret: List[InputData] = []
        for data_point in data:
            prompt = (
                "Please answer the following question with true or false: "
                + f"{data_point['question']}?\nAnswer:"
            )
            answer = "true" if data_point["answer"] else "false"
            if is_train:
                prompt += f" {answer}"
                labels = None
            else:
                labels = [self.labels2id_[answer]]
            ret.append(InputData(inputs=prompt, labels=labels))

        return ret


class OpenBookQA(QuestionAnswerTask):
    def __init__(self) -> None:
        super().__init__(["A", "B", "C", "D"])

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        data = hf_datasets.load_dataset(
            "allenai/openbookqa" if path is None else path, "main"
        )["train" if is_train else "test"]
        logging.info("Preparing data for OpenBookQA")
        ret: List[InputData] = []
        for data_point in data:
            prompt = (
                "Please choose the correct answer to the question: "
                + data_point["question_stem"]
            )
            choices = data_point["choices"]
            for label, text in zip(choices["label"], choices["text"]):
                prompt += f" ({label}) {text}"
            prompt += "\nAnswer:"
            if is_train:
                prompt += " " + data_point["answerKey"]
                labels = None
            else:
                labels = [self.labels2id_[data_point["answerKey"]]]
            ret.append(InputData(inputs=prompt, labels=labels))

        return ret


class PIQA(QuestionAnswerTask):
    def __init__(self) -> None:
        super().__init__(["A", "B"])

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        data = hf_datasets.load_dataset("piqa" if path is None else path)[
            "train" if is_train else "validation"
        ]
        logging.info("Preparing data for PIQA")
        ret: List[InputData] = []
        for data_point in data:
            prompt = "Below is a common task along with two possible solutions labeled (A) and (B)."
            prompt += f" Please select the appropriate solution to achieve the task:\n{data_point['goal']}\n"
            prompt += f"\n(A) {data_point['sol1']}\n(B) {data_point['sol2']}\n"
            prompt += "\nCorrect solution:"
            answer = self.labels_[data_point["label"]]
            if is_train:
                prompt += f" {answer}"
                labels = None
            else:
                labels = [data_point["label"]]
            ret.append(InputData(inputs=prompt, labels=labels))

        return ret


class SIQA(QuestionAnswerTask):
    def __init__(self) -> None:
        super().__init__(["A", "B", "C"])

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        data = hf_datasets.load_dataset("social_i_qa" if path is None else path)[
            "train" if is_train else "validation"
        ]
        logging.info("Preparing data for SIQA")
        ret: List[InputData] = []
        for data_point in data:
            prompt = "Please choose the correct answer to the question.\n"
            prompt += f"Question: {data_point['context']} {data_point['question']}"
            prompt += f"\n(A) {data_point['answerA']}"
            prompt += f"\n(B) {data_point['answerB']}"
            prompt += f"\n(C) {data_point['answerC']}"
            prompt += "\nAnswer:"
            label = int(data_point["label"]) - 1
            if is_train:
                prompt += f" {self.labels_[label]}"
                labels = None
            else:
                labels = [label]
            ret.append(InputData(inputs=prompt, labels=labels))

        return ret


class HellaSwag(QuestionAnswerTask):
    def __init__(self) -> None:
        super().__init__(["A", "B", "C", "D"])

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        data = hf_datasets.load_dataset("Rowan/hellaswag" if path is None else path)[
            "train" if is_train else "validation"
        ]
        logging.info("Preparing data for HellaSwag")
        ret: List[InputData] = []
        for data_point in data:
            prompt = (
                "Please choose the correct ending to complete the given sentence.\n"
            )
            prompt += f"Sentence: {data_point['activity_label']}. {data_point['ctx']}"
            for label, text in enumerate(data_point["endings"]):
                prompt += f"\n({self.labels_[label]}) {text}"
            prompt += "\nAnswer:"
            label = int(data_point["label"])
            if is_train:
                prompt += f" {self.labels_[label]}"
                labels = None
            else:
                labels = [label]
            ret.append(InputData(inputs=prompt, labels=labels))

        return ret


class WinoGrande(QuestionAnswerTask):
    def __init__(self) -> None:
        super().__init__(["A", "B"])

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        data = hf_datasets.load_dataset(
            "winogrande" if path is None else path, "winogrande_debiased"
        )["train" if is_train else "validation"]
        logging.info("Preparing data for WinoGrande")
        ret: List[InputData] = []
        for data_point in data:
            prompt = "Please choose the correct answer to fill in the blank to complete the given sentence.\n"
            prompt += f"Sentence: {data_point['sentence']}"
            prompt += f"\n(A) {data_point['option1']}\n(B) {data_point['option2']}"
            prompt += "\nAnswer:"
            label = int(data_point["answer"]) - 1
            if is_train:
                prompt += f" {self.labels_[label]}"
                labels = None
            else:
                labels = [label]
            ret.append(InputData(inputs=prompt, labels=labels))

        return ret


class CommonSenseQA(QuestionAnswerTask):
    def __init__(self) -> None:
        super().__init__(["A", "B", "C", "D", "E"])

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        data = hf_datasets.load_dataset("tau/commonsense_qa" if path is None else path)[
            "train" if is_train else "validation"
        ]
        logging.info("Preparing data for CommonSenseQA")
        ret: List[InputData] = []
        for data_point in data:
            prompt = (
                "Please choose the correct answer to the question: "
                + data_point["question"]
            )
            choices = data_point["choices"]
            for label, text in zip(choices["label"], choices["text"]):
                prompt += f" ({label}) {text}"
            prompt += "\nAnswer:"
            if is_train:
                prompt += " " + data_point["answerKey"]
                labels = None
            else:
                labels = [self.labels2id_[data_point["answerKey"]]]
            ret.append(InputData(inputs=prompt, labels=labels))

        return ret


class PubMedQA(QuestionAnswerTask):
    def __init__(self) -> None:
        super().__init__(["yes", "no", "maybe"])

    def loading_data(
        self, is_train: bool = True, path: Optional[str] = None
    ) -> List[InputData]:
        data = hf_datasets.load_dataset(
            "qiaojin/PubMedQA" if path is None else path,
            "pqa_artificial" if is_train else "pqa_labeled",
        )["train"]
        logging.info("Preparing data for PubMedQA")
        ret: List[InputData] = []
        for data_point in data:
            prompt = (
                "Instruction:\nPlease answer the following question with yes or no "
                + "based on your medical knowledge and the following context.\n"
                + f"Question:\n{data_point['question']}\nContext:\n"
            )
            context = data_point["context"]
            for label, text in zip(context["labels"], context["contexts"]):
                prompt += f"({label}) {text}\n"
            answer = data_point["final_decision"]
            assert answer in self.labels2id_
            if is_train:
                prompt += f"Long Answer:\n{data_point['long_answer']}\n"
                prompt += "Answer:"
                prompt += f" {answer}"
                labels = None
            else:
                prompt += "Answer:"
                labels = [self.labels2id_[answer]]
            ret.append(InputData(inputs=prompt, labels=labels))

        return ret


def update_task_dict(task_dict):
    task_dict.update(
        {
            "arc-e": ARC("ARC-Easy"),
            "arc-c": ARC("ARC-Challenge"),
            "boolq": BoolQ(),
            "obqa": OpenBookQA(),
            "piqa": PIQA(),
            "siqa": SIQA(),
            "hellaswag": HellaSwag(),
            "winogrande": WinoGrande(),
            "csqa": CommonSenseQA(),
            "pubmedqa": PubMedQA(),
        }
    )
