from mlora.config import TaskConfig
from mlora.prompter import Prompter
from mlora.model.args import LinearInfo, Tokens, Masks
from mlora.model.tokenizer import Tokenizer
from mlora.trainer.context import TaskContext, TASKCONTEXT_CLASS

import os
import json
import torch
import logging
from tqdm import tqdm
from datasets import load_dataset
from collections import OrderedDict
from typing import Dict, Callable, List, Optional, Tuple


class Task:
    config_: TaskConfig

    now_step_: int = 0
    now_epoch_: int = 0

    tokenizer_: Tokenizer = None
    context_: TaskContext = None

    train_data_: List[Dict[str, str]] = None
    now_data_idx_: int = 0

    prompter_: Prompter = None

    llm_name_: str = ""

    def __init__(self, config: TaskConfig, llm_name: str) -> None:
        self.config_ = config
        self.tokenizer_ = None
        self.context_ = None

        self.now_step_ = 1
        self.now_epoch_ = 1
        self.now_data_idx_ = 0
        self.train_data_ = []

        self.prompter_ = Prompter(config.dataset_.prompt_path_)
        self.llm_name_ = llm_name

    def is_done(self) -> bool:
        if self.now_epoch_ <= self.config_.num_epochs_:
            return False
        return True

    def __pre_dataset(self):
        preprocess_func: Dict[str, Callable] = {
            "default": lambda data: data,
            "shuffle": lambda data: data.shuffle(),
            "sort": lambda data: data.sort()
        }

        logging.info(f"Task - {self.config_.adapter_.name_} load data")
        data = load_dataset(
            "json", data_files=self.config_.dataset_.data_path_)

        preprocess_type = self.config_.dataset_.preprocess_
        if preprocess_type not in preprocess_func:
            raise NotImplementedError

        data = preprocess_func[preprocess_type](data)
        logging.info(
            f'Task - {self.config_.adapter_.name_} data size: {len(data["train"])} '
            f'epoch: {self.config_.num_epochs_} batch size: {self.config_.batch_size_} / {self.config_.mini_batch_size_}')

        for _, data_point in tqdm(enumerate(data["train"])):
            self.train_data_.append(data_point)

    def __pre_context(self, linears_info: OrderedDict[str, LinearInfo]):
        self.context_ = TASKCONTEXT_CLASS[self.config_.adapter_.type_](
            self.config_.adapter_.name_)

        self.context_.init_adapter(self.config_.adapter_, linears_info)

    def pre_init(self, linears_info: OrderedDict[str, LinearInfo], tokenizer: Tokenizer):
        self.tokenizer_ = tokenizer
        # prepare the dataset and context
        self.__pre_dataset()
        self.__pre_context(linears_info)

    def switch_device(self, device: str):
        self.context_.switch_device(device)

    def get_train_data(self) -> List[Tokens]:
        logging.info(
            f'Task - {self.context_.name_} epoch: {self.now_epoch_}/{self.config_.num_epochs_}'
            f' iteration: {self.now_data_idx_}/{len(self.train_data_)} step: {self.now_step_}')
        data_idx_s = self.now_data_idx_
        data_idx_e = self.now_data_idx_ + self.config_.mini_batch_size_

        batch_str = self.prompter_.generate_prompt_batch(
            self.train_data_[data_idx_s:data_idx_e])

        return list(map(lambda raw_str: self.tokenizer_.encode(raw_str, bos=True, eos=True), batch_str))

    def get_loss_fn(self) -> torch.nn.Module:
        return self.context_.loss_fn_

    def expand_batch_tokens(self, batch_tokens: List[Tokens],
                            align_len: Optional[int] = None) -> Tuple[List[Tokens], List[Masks]]:
        if align_len is None:
            align_len = max(map(lambda x: len(x), batch_tokens))

        ret_batch_tokens = []
        ret_batch_masks = []
        for tokens in batch_tokens:
            tokens, masks = self.tokenizer_.expand_tokens(tokens, align_len)
            ret_batch_tokens.append(tokens)
            ret_batch_masks.append(masks)

        return ret_batch_tokens, ret_batch_masks

    def save(self, dir_suffix: str = "", additional_info: Dict[str, str] = {}):
        output_dir = self.context_.name_
        if dir_suffix != "":
            output_dir += os.sep + self.context_.name_ + "_" + dir_suffix

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        torch.save(self.context_.weight_dict(),
                   output_dir + os.sep + "adapter_model.bin")

        adapter_config: Dict[str, str] = {}
        adapter_config["base_model_name_or_path"] = self.llm_name_
        adapter_config = {**adapter_config, **additional_info}
        adapter_config = {**adapter_config, **self.config_.adapter_.export()}

        with open(output_dir + os.sep + "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=4)

    def step(self):
        stepd: bool = False

        if self.now_step_ % self.config_.accumulate_step == 0:
            stepd = True
            self.context_.step()

        # to save the model
        if self.now_step_ % self.config_.save_step_ == 0:
            self.save(f"{self.now_step_}")

        self.now_step_ += 1
        self.now_data_idx_ += self.config_.mini_batch_size_

        if self.now_data_idx_ >= len(self.train_data_):
            self.now_epoch_ += 1
            self.now_data_idx_ = 0

        # task finish we also need to step
        if not stepd and self.now_epoch_ >= self.config_.num_epochs_:
            self.context_.step()
