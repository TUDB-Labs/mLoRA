from mlora.model.llm.model import LLMModel
from mlora.model.args import Tokens, MLoRABatchData
from mlora.model.tokenizer.tokenizer import Tokenizer
from mlora.evaluator.evaluator_factory import Evaluator

import math
import torch
import datasets
import logging

from typing import List, Tuple


class MMLUEvaluator(Evaluator):
    data_: str = ""
    choices_map_: List[str] = None
    choices_map_tokens_: List[int] = None
    subject_: str = "all"
    kshot_: int = 5
    max_seq_: int = 2048
    batch_size_: int = 2

    def parse_arguments(self, args: List[str]):
        # get arguments from the args mmlu:<subject>:[kshot]:[batch_size]:[max_seq]
        IDX_SUBJECT = 0
        IDX_KSHOT = 1
        IDX_BATCH_SIZE = 2
        IDX_MAX_SEQ_LEN = 3

        if len(args) >= 1 and args[IDX_SUBJECT] != "":
            self.subject_ = args[IDX_SUBJECT]

        if len(args) >= 2 and args[IDX_KSHOT] != "":
            assert args[IDX_KSHOT].isdigit(
            ), f"argument error {args[IDX_KSHOT]} must digit."
            self.kshot_ = int(args[IDX_KSHOT])

        if len(args) >= 3 and args[IDX_BATCH_SIZE] != "":
            assert args[IDX_BATCH_SIZE].isdigit(
            ), f"argument error {args[IDX_BATCH_SIZE]} must digit."
            self.batch_size_ = int(args[IDX_BATCH_SIZE])

        if len(args) >= 4 and args[IDX_MAX_SEQ_LEN] != "":
            assert args[IDX_MAX_SEQ_LEN].isdigit(
            ), f"argument error {args[IDX_MAX_SEQ_LEN]} must digit."
            self.max_seq_ = int(args[IDX_MAX_SEQ_LEN])

    def __init__(self,
                 model: LLMModel,
                 tokenizer: Tokenizer,
                 data: str,
                 args: List[str]):
        super().__init__()
        # data_: the path or name of mmlu datasets
        self.parse_arguments(args)

        self.model_ = model
        self.tokenizer_ = tokenizer
        self.data_ = data

        self.choices_map_ = ["A", "B", "C", "D"]
        self.choices_map_tokens_ = [self.tokenizer_.encode(
            choice, bos=False, eos=False)[0] for choice in self.choices_map_]

    def prepare_evaluate_data(self, subject: str) -> Tuple[List[Tokens], List[str]]:
        # return val: the tokens and the labels
        mmlu_data_set = datasets.load_dataset(self.data_, subject)
        dev_data = mmlu_data_set["dev"]
        test_data = mmlu_data_set["test"]

        def format_subject(subject: str):
            return subject.replace("_", " ")

        def format_prompt(data_point, with_answer=True):
            # get the question and choices like:
            #   <QUESTION>
            #   A. <CHOICEA>
            #   B. <CHOICEB>
            #   ....
            #   Answer:
            choices_list = [f"{key}. {choice}\n" for key,
                            choice in zip(self.choices_map_, data_point["choices"])]

            question = data_point["question"].strip()
            choices = "".join(choices_list)

            prompt = f"{question}\n{choices}Answer:"

            if with_answer:
                prompt += " {}\n\n".format(
                    self.choices_map_[data_point["answer"]])

            return prompt

        all_tokens: List[Tokens] = []
        all_labels: List[str] = []

        for test_data_point in test_data:
            kshot_prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
                format_subject(subject))
            test_prompt = format_prompt(test_data_point, False)

            tokens: Tokens = []

            # prepare kshot case
            # kshot_prompt + [dev_prompt] + test_prompt, see format_prompt function
            for (dev_shot_idx, dev_data_point) in enumerate(dev_data):
                if dev_shot_idx >= self.kshot_:
                    break
                # prepare the prompt, use k-show
                dev_prompt = format_prompt(dev_data_point, True)
                tmp_tokens = self.tokenizer_.encode(
                    kshot_prompt + dev_prompt + test_prompt, bos=True, eos=False)
                if len(tmp_tokens) > self.max_seq_:
                    break
                # to ensure the last one is eos token
                if len(tmp_tokens) == self.max_seq_:
                    tmp_tokens[-1] = self.tokenizer_.eos_id_
                tokens = tmp_tokens
                kshot_prompt += dev_prompt

            assert len(tokens) > 0
            all_tokens.append(tokens)
            all_labels.append(self.choices_map_[test_data_point["answer"]])

        return all_tokens, all_labels

    def get_choice_from_output(self, prob: torch.Tensor) -> str:
        choice_prob = torch.tensor([prob[choice_token]
                                    for choice_token in self.choices_map_tokens_])
        max_prob = torch.argmax(choice_prob).item()
        return self.choices_map_[max_prob]

    def model_inference(self, tokens: List[Tokens]) -> List[str]:

        def pad_to_len(wait_to_pad: Tokens, seq_len: int):
            while len(wait_to_pad) < seq_len:
                wait_to_pad.append(self.tokenizer_.pad_id_)
            return wait_to_pad

        choice_probs: List[str] = []

        for start_pos in range(0, len(tokens), self.batch_size_):
            batch_data = tokens[start_pos:start_pos + self.batch_size_]
            # bd short for batch data
            bd_tokens_len = [len(token) for token in batch_data]
            # get max seq len and align with 8
            to_pad_len = max(bd_tokens_len)
            to_pad_len = math.ceil(to_pad_len / 8) * 8
            # pad to it
            aligned_batch_data = [pad_to_len(token, to_pad_len)
                                  for token in batch_data]
            # generate the pad
            attention_mask = [self.tokenizer_.mask_from(
                token) for token in aligned_batch_data]

            # TODO lora model
            output: torch.Tensor = self.model_.forward(MLoRABatchData(batch_tokens_=aligned_batch_data,
                                                                      batch_mask_=attention_mask,
                                                                      lora_batch_data_config_=None,
                                                                      inference_model_=True))
            # only get the last predict value
            output = [output[idx][len - 1]
                      for idx, len in enumerate(bd_tokens_len)]
            # get the choice
            choice_probs.extend([self.get_choice_from_output(
                each_output)for each_output in output])

            del output

        return choice_probs

    def evaluate_subject(self, subject: str) -> List[bool]:
        tokens, labels = self.prepare_evaluate_data(subject)
        choices: List[str] = self.model_inference(tokens)
        assert len(choices) == len(labels)

        result: List[bool] = []
        for idx in range(0, len(choices)):
            result.append(labels[idx] == choices[idx])

        return result

    @torch.inference_mode()
    def evaluate(self) -> float:
        logging.info(f"Performing MMLU/{self.subject_} Benchmark.")
        result = self.evaluate_subject(self.subject_)
        logging.info(f"Accuracy: {sum(result)/len(result)}")
