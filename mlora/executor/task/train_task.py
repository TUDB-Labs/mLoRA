from mlora.config import TaskConfig
from mlora.model.args import LinearInfo, Tokens, Masks, MLoRADataConfig
from mlora.model.tokenizer import Tokenizer

import os
import json
import torch
import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, override

from .task import Task


class TrainTask(Task):
    now_epoch_: int = 0

    def __init__(self, config: TaskConfig, llm_name: str) -> None:
        super().__init__(config, llm_name)
        self.now_epoch_ = 1

    @override
    def is_done(self) -> bool:
        return self.now_epoch_ > self.config_.num_epochs_

    @override
    def prepare(self, linears_info: OrderedDict[str, LinearInfo], tokenizer: Tokenizer):
        self.tokenizer_ = tokenizer
        # prepare the dataset and context
        self._pre_dataset()
        self._pre_context(linears_info)

    @override
    def data(self, start_idx: int) -> Tuple[List[Tokens], List[MLoRADataConfig]]:
        logging.info(
            f'Adapter {self.context_.name_} epoch: {
                self.now_epoch_}/{self.config_.num_epochs_}'
            f' iteration: {self.now_data_idx_}/{len(self.data_)} step: {self.now_step_}')
        data_idx_s = self.now_data_idx_
        data_idx_e = self.now_data_idx_ + self.config_.mini_batch_size_

        # get the train raw string
        batch_str = self.prompter_.generate_prompt_batch(
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

        torch.save(self.context_.weight_dict(),
                   output_dir + os.sep + "adapter_model.bin")

        adapter_config: Dict[str, str] = {}
        adapter_config["base_model_name_or_path"] = self.llm_name_
        adapter_config = {**adapter_config, **additional_info}
        adapter_config = {**adapter_config, **self.config_.adapter_.export()}

        with open(output_dir + os.sep + "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=4)

    @override
    def done(self):
        self._save()
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
            self._save(f"{self.now_step_}")

        self.now_step_ += 1
        self.now_data_idx_ += self.config_.mini_batch_size_

        if self.now_data_idx_ >= len(self.data_):
            self.now_epoch_ += 1
            self.now_data_idx_ = 0

        # task finish we also need to step
        if not stepd and self.now_epoch_ >= self.config_.num_epochs_:
            self.context_.step()
