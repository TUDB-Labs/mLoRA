from .common import BasicMetric, AutoMetric, CommonSenseTask
from mlora.tokenizer import Tokenizer
from mlora.common import DataClass
from typing import List

import datasets as hf_datasets
import logging
import torch


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

    def loading_data(self,
                     tokenizer: Tokenizer,
                     is_train: bool = True) -> List[DataClass]:
        data = hf_datasets.load_dataset(
            "allenai/ai2_arc", self.subject_)["train" if is_train else "test"]
        logging.info(f"Preparing data for {self.subject_}")
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
            tokens = tokenizer.encode(data=prompt)
            ret.append(DataClass(tokens_=tokens, labels_=labels))
            if idx % 10000 == 0:
                logging.info(f"Encode text data: {idx}/{len(data)}")

        return ret


class BoolQ(QuestionAnswerTask):
    def __init__(self) -> None:
        super().__init__(["true", "false"])

    def loading_data(self,
                     tokenizer: Tokenizer,
                     is_train: bool = True) -> List[DataClass]:
        data = hf_datasets.load_dataset(
            "google/boolq")["train" if is_train else "validation"]
        logging.info("Preparing data for BoolQ")
        ret: List[DataClass] = []
        for idx, data_point in enumerate(data):
            prompt = "Please answer the following question with true or false: " + \
                f"{data_point['question']}?\nAnswer:"
            answer = "true" if data_point["answer"] else "false"
            if is_train:
                prompt += f" {answer}"
                labels = None
            else:
                labels = [self.labels2id_[answer]]
            tokens = tokenizer.encode(data=prompt)
            ret.append(DataClass(tokens_=tokens, labels_=labels))
            if idx % 10000 == 0:
                logging.info(f"Encode text data: {idx}/{len(data)}")

        return ret


class OpenBookQA(QuestionAnswerTask):
    def __init__(self) -> None:
        super().__init__(["A", "B", "C", "D"])

    def loading_data(self,
                     tokenizer: Tokenizer,
                     is_train: bool = True) -> List[DataClass]:
        data = hf_datasets.load_dataset(
            "allenai/openbookqa", "main")["train" if is_train else "test"]
        logging.info("Preparing data for OpenBookQA")
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
            tokens = tokenizer.encode(data=prompt)
            ret.append(DataClass(tokens_=tokens, labels_=labels))
            if idx % 10000 == 0:
                logging.info(f"Encode text data: {idx}/{len(data)}")

        return ret


class PIQA(QuestionAnswerTask):
    def __init__(self) -> None:
        super().__init__(["A", "B"])

    def loading_data(self,
                     tokenizer: Tokenizer,
                     is_train: bool = True) -> List[DataClass]:
        data = hf_datasets.load_dataset(
            "piqa")["train" if is_train else "validation"]
        logging.info("Preparing data for PIQA")
        ret: List[DataClass] = []
        for idx, data_point in enumerate(data):
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
            tokens = tokenizer.encode(data=prompt)
            ret.append(DataClass(tokens_=tokens, labels_=labels))
            if idx % 10000 == 0:
                logging.info(f"Encode text data: {idx}/{len(data)}")

        return ret


class SIQA(QuestionAnswerTask):
    def __init__(self) -> None:
        super().__init__(["A", "B", "C"])

    def loading_data(self,
                     tokenizer: Tokenizer,
                     is_train: bool = True) -> List[DataClass]:
        data = hf_datasets.load_dataset(
            "social_i_qa")["train" if is_train else "validation"]
        logging.info("Preparing data for SIQA")
        ret: List[DataClass] = []
        for idx, data_point in enumerate(data):
            prompt = "Please choose the correct answer to the question.\n"
            prompt += f"Question: {data_point['context']} {data_point['question']}"
            prompt += f"\n(A) {data_point['answerA']}"
            prompt += f"\n(B) {data_point['answerB']}"
            prompt += f"\n(C) {data_point['answerC']}"
            prompt += "\nAnswer:"
            label = int(data_point['label']) - 1
            if is_train:
                prompt += f" {self.labels_[label]}"
                labels = None
            else:
                labels = [label]
            tokens = tokenizer.encode(data=prompt)
            ret.append(DataClass(tokens_=tokens, labels_=labels))
            if idx % 10000 == 0:
                logging.info(f"Encode text data: {idx}/{len(data)}")

        return ret


class HellaSwag(QuestionAnswerTask):
    def __init__(self) -> None:
        super().__init__(["A", "B", "C", "D"])

    def loading_data(self,
                     tokenizer: Tokenizer,
                     is_train: bool = True) -> List[DataClass]:
        data = hf_datasets.load_dataset(
            "Rowan/hellaswag")["train" if is_train else "validation"]
        logging.info("Preparing data for HellaSwag")
        ret: List[DataClass] = []
        for idx, data_point in enumerate(data):
            prompt = "Please choose the correct ending to complete the given sentence.\n"
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
            tokens = tokenizer.encode(data=prompt)
            ret.append(DataClass(tokens_=tokens, labels_=labels))
            if idx % 10000 == 0:
                logging.info(f"Encode text data: {idx}/{len(data)}")

        return ret


class WinoGrande(QuestionAnswerTask):
    def __init__(self) -> None:
        super().__init__(["A", "B"])

    def loading_data(self,
                     tokenizer: Tokenizer,
                     is_train: bool = True) -> List[DataClass]:
        data = hf_datasets.load_dataset(
            "winogrande", "winogrande_debiased")["train" if is_train else "validation"]
        logging.info("Preparing data for WinoGrande")
        ret: List[DataClass] = []
        for idx, data_point in enumerate(data):
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
            tokens = tokenizer.encode(data=prompt)
            ret.append(DataClass(tokens_=tokens, labels_=labels))
            if idx % 10000 == 0:
                logging.info(f"Encode text data: {idx}/{len(data)}")

        return ret


class MedicalQA(QuestionAnswerTask):
    def __init__(self) -> None:
        super().__init__(["Yes", "No"])

    def loading_data(self,
                     tokenizer: Tokenizer,
                     is_train: bool = True) -> List[DataClass]:
        data = hf_datasets.load_dataset(
            "TUDB-Labs/medical-qa")["train" if is_train else "test"]
        logging.info("Preparing data for MedicalQA")
        ret: List[DataClass] = []
        for idx, data_point in enumerate(data):
            prompt = "Please provide your answer to the following question using" \
                + " information from medical articles and your medical expertise.\n"
            prompt += f"Information: {data_point['short_context' if is_train else 'long_context']}\n"
            prompt += f"Question: {data_point['question']}\n"
            prompt += "Answer:"
            answer = data_point["raw_answer"]
            if is_train:
                prompt += f" {answer}"
                labels = None
            else:
                labels = [self.labels2id_[answer]]
            tokens = tokenizer.encode(data=prompt)
            ret.append(DataClass(tokens_=tokens, labels_=labels))
            if idx % 10000 == 0:
                logging.info(f"Encode text data: {idx}/{len(data)}")

        return ret


def update_task_dict(task_dict):
    task_dict.update({
        "arc-e": ARC("ARC-Easy"),
        "arc-c": ARC("ARC-Challenge"),
        "boolq": BoolQ(),
        "obqa": OpenBookQA(),
        "piqa": PIQA(),
        "siqa": SIQA(),
        "hellaswag": HellaSwag(),
        "winogrande": WinoGrande(),
        "medical-qa": MedicalQA(),
    })
