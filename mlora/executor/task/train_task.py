import json
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, override

import torch

from mlora.config import TaskConfig, TrainTaskConfig
from mlora.executor.context import TrainTaskContext
from mlora.model.args import LinearInfo, Masks, MLoRADataConfig, Tokens
from mlora.model.tokenizer import Tokenizer

from .task import Task


def _parse_dir_name(dir_name: str) -> Tuple[int, int, int]:
    split_group = dir_name.split("_")

    epoch = int(split_group[1]) if len(split_group) >= 2 else None
    data_idx = int(split_group[2]) if len(split_group) >= 3 else None
    step = int(split_group[3]) if len(split_group) >= 4 else None

    return epoch, data_idx, step


def _list_folders_in_dir(directory: str) -> List[str]:
    ret_folders = list(
        filter(
            lambda name: os.path.isdir(os.path.join(directory, name)),
            os.listdir(directory),
        )
    )
    return [os.path.join(directory, folder) for folder in ret_folders]


class TrainTask(Task):
    now_epoch_: int
    context_: TrainTaskContext
    config_: TrainTaskConfig
    recover_folder: str

    def __init__(self, config: TaskConfig, llm_name: str) -> None:
        super().__init__(config, llm_name)

    @override
    def is_done(self) -> bool:
        return self.now_epoch_ > self.config_.num_epochs_

    @override
    def prepare(self, linears_info: OrderedDict[str, LinearInfo], tokenizer: Tokenizer):
        self.tokenizer_ = tokenizer
        # prepare the context and the dataset
        # NOTE: how to recover the sort of dataset
        self._pre_dataset()
        self._pre_context(linears_info)
        self._pre_recover_state()
        self._pre_recover_context()

    def _pre_recover_state(self):
        recover_folders = _list_folders_in_dir(
            os.path.dirname(self.config_.adapter_.path_)
        )
        recover_folders = list(
            filter(
                lambda folder: self.config_.adapter_.path_ in folder, recover_folders
            )
        )
        if recover_folders is None or len(recover_folders) <= 0:
            self.now_epoch_ = 1
            return

        max_step = -1
        for folder in recover_folders:
            base_folder = os.path.basename(os.path.normpath(folder))
            epoch, data_idx, step = _parse_dir_name(base_folder)
            if step is not None and step > max_step:
                max_step = max(max_step, step)
                self.now_epoch_ = epoch + 1
                self.now_data_idx_ = data_idx + 1
                self.now_step_ = step + 1

    def _pre_recover_context(self):
        if self.now_step_ <= 1:
            return

        # get the optimizer read the file from now_epoch
        checkpoint = torch.load(self.recover_folder + os.sep + "checkpoint.bin")

        self.context_.recover_weight(checkpoint["weight_dict"])
        self.context_.recover_optimizer(checkpoint["state_dict"])
        self.context_.recover_lr(self.now_epoch_)

    @override
    def data(self, start_idx: int) -> Tuple[List[Tokens], List[MLoRADataConfig]]:
        logging.info(
            f"Adapter {self.context_.name_} "
            f"epoch: {self.now_epoch_}/{self.config_.num_epochs_} "
            f"iteration: {self.now_data_idx_}/{len(self.data_)} step: {self.now_step_}"
        )
        data_idx_s = self.now_data_idx_
        data_idx_e = self.now_data_idx_ + self.config_.mini_batch_size_

        # get the train raw string
        batch_str = self.prompter_.generate_prompt(self.data_[data_idx_s:data_idx_e])

        # convert the string to tokens
        ret_tokens = list(
            map(
                lambda raw_str: self.tokenizer_.encode(
                    raw_str, bos=True, eos=True, cutoff_len=self.config_.cutoff_len_
                ),
                batch_str,
            )
        )
        end_idx = start_idx + len(ret_tokens)

        def loss_fn(
            input: torch.Tensor, target: torch.Tensor, _: torch.Tensor
        ) -> torch.Tensor:
            vacab_size = input.shape[-1]
            loss_input = (
                input[start_idx:end_idx, :-1, :].contiguous().view(-1, vacab_size)
            )
            loss_target = (
                target[start_idx:end_idx, 1:]
                .contiguous()
                .view(-1)
                .to(loss_input.device)
            )
            loss = self.context_.loss_fn_(loss_input, loss_target)

            logging.info(f"Adapter {self.context_.name_} loss: {loss}")

            return loss

        data_config = MLoRADataConfig(
            self.context_.name_,
            self.context_.type_,
            start_idx,
            end_idx,
            self._expand_batch_tokens,
            loss_fn,
        )

        return ret_tokens, [data_config]

    def _expand_batch_tokens(
        self, batch_tokens: List[Tokens], align_len: Optional[int] = None
    ) -> Tuple[List[Tokens], List[Masks]]:
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
            output_dir = (
                self.context_.path_
                + "_"
                + "_".join([str(self.now_epoch_), str(self.now_data_idx_), dir_suffix])
            )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        weight_dict = self.context_.weight_dict()
        state_dict = self.context_.state_dict()
        # save to disk
        checkpoint = {"weight_dict": weight_dict, "state_dict": state_dict}
        torch.save(checkpoint, output_dir + os.sep + "checkpoint.bin")

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
    def terminate(self):
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

    @override
    def task_progress(self) -> int:
        total_step = len(self.data_) // self.config_.mini_batch_size_
        total_step = total_step * self.config_.num_epochs_
        return int((self.now_step_ / total_step) * 100)
