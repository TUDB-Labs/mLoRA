from mlora.config import TaskConfig
from mlora.model.args import LinearInfo, Tokens, Masks, MLoRADataConfig
from mlora.model.tokenizer import Tokenizer
import re
import os
import json
import torch
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, override

from .task import Task


class TrainTask(Task):
    now_epoch_: int = 0
    is_restore : bool = False
    checkpoint = None
    def __init__(self, config: TaskConfig, llm_name: str) -> None:
        super().__init__(config, llm_name)
        self.restore()
        if self.is_restore :

            self.now_epoch_ = self.checkpoint["epoch"]+1
        else :
            self.now_epoch_ = 1

    @override
    def is_done(self) -> bool:
        return self.now_epoch_ > self.config_.num_epochs_

    @override
    def prepare(self, linears_info: OrderedDict[str, LinearInfo], tokenizer: Tokenizer):
        self.tokenizer_ = tokenizer
        # prepare the dataset and context
        self._pre_dataset()
        self._pre_context(linears_info,self.checkpoint)

    @override
    def data(self, start_idx: int) -> Tuple[List[Tokens], List[MLoRADataConfig]]:
        logging.info(
            f'Adapter {self.context_.name_} epoch: {
                self.now_epoch_}/{self.config_.num_epochs_}'
            f' iteration: {self.now_data_idx_}/{len(self.data_)} step: {self.now_step_}')
        data_idx_s = self.now_data_idx_
        data_idx_e = self.now_data_idx_ + self.config_.mini_batch_size_

        # get the train raw string
        batch_str = self.prompter_.generate_prompt(
            self.data_[data_idx_s:data_idx_e])

        # convert the string to tokens
        ret_tokens = list(map(lambda raw_str: self.tokenizer_.encode(
            raw_str, bos=True, eos=True), batch_str))
        end_idx = start_idx + len(ret_tokens)

        def loss_fn(input: torch.Tensor, target: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
            vacab_size = input.shape[-1]
            loss_input = input[start_idx:end_idx, :-1,
                               :].contiguous().view(-1, vacab_size)
            loss_target = target[start_idx:end_idx,
                                 1:].contiguous().view(-1).to(loss_input.device)
            loss = self.context_.loss_fn_(loss_input, loss_target)

            logging.info(f"Adapter {self.context_.name_} loss: {loss}")

            return loss

        data_config = MLoRADataConfig(self.context_.name_, self.context_.type_,
                                      start_idx, end_idx,
                                      self._expand_batch_tokens, loss_fn)

        return ret_tokens, [data_config]

    def _expand_batch_tokens(self,
                             batch_tokens: List[Tokens],
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

    def _save(self, dir_suffix: str = "", additional_info: Dict[str, str] = {}):
        output_dir = self.context_.path_
        if dir_suffix != "":
            output_dir += os.sep + self.context_.path_ + "_" + dir_suffix

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        temp_dic = self.context_.checkpoint()
        temp_dic["epoch"] = self.now_epoch_
        logging.info(f"save checkpoint in {output_dir + os.sep}checkpoint.bin")
        torch.save(temp_dic,
                   output_dir + os.sep + "checkpoint.bin")

        adapter_config: Dict[str, str] = {}
        adapter_config["base_model_name_or_path"] = self.llm_name_
        adapter_config = {**adapter_config, **additional_info}
        adapter_config = {**adapter_config, **self.config_.adapter_.export()}

        with open(output_dir + os.sep + "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=4)

    @override
    def done(self):
        self._save(f"{self.now_epoch_}")
        # release the context
        del self.context_

    @override
    def step(self):
        stepd: bool = False

        if self.now_step_ % self.config_.accumulate_step_ == 0:
            stepd = True
            self.context_.step()

        # to save the model
        if self.now_step_ % self.config_.save_step_ == 0:
            self._save(f"{self.now_epoch_}")

        self.now_step_ += 1
        self.now_data_idx_ += self.config_.mini_batch_size_

        if self.now_data_idx_ >= len(self.data_):
            self.now_epoch_ += 1
            self.now_data_idx_ = 0

        # task finish we also need to step
        if not stepd and self.now_epoch_ >= self.config_.num_epochs_:
            self.context_.step()

    def restore(self):
        temp_path = self.config_.adapter_.path_
        if os.path.isdir(os.path.join(temp_path, "adapters")):
            self.is_restore = True
            temp_path = os.path.join(temp_path, "adapters")
            folders = [folder for folder in os.listdir(temp_path)]
            max_suffix = 0
            max_dir = None
            for dir_path in folders:
                suffix = self.extract_dir_suffix(dir_path)
                if suffix is not None and suffix > max_suffix:
                    max_suffix = suffix
                    max_dir = dir_path
            temp_path=temp_path+os.sep+max_dir
            logging.info(f"load checkpoint in {temp_path+os.sep}checkpoint.bin")
            self.checkpoint = torch.load(temp_path+os.sep+"checkpoint.bin")
    def extract_dir_suffix(self,path):
        match = re.search(r'_(\d+)$', path)
        return int(match.group(1)) if match else None