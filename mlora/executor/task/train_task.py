import json
import logging
import os
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, override

import torch

from mlora.config import TaskConfig, TrainTaskConfig
from mlora.executor.context import TrainTaskContext
from mlora.model.args import LinearInfo, Masks, MLoRADataConfig, Tokens
from mlora.model.tokenizer import Tokenizer

from .task import Task

def parse_output_dir(output_dir, context_path):
    if output_dir == context_path:
        return {
            'is_suffixed': False,
            'epoch': None,
            'data_idx': None,
            'dir_suffix': None
        }

    suffix_pattern = re.compile(r'_(?P<epoch>\d+)_(?P<data_idx>\d+)_(?P<dir_suffix>[^_]+)$')
    match = suffix_pattern.search(output_dir)

    if not match:
        raise ValueError("output_dir does not match the expected format.")

    epoch = int(match.group('epoch'))
    data_idx = int(match.group('data_idx'))
    dir_suffix = match.group('dir_suffix')

    return {
        'is_suffixed': True,
        'epoch': epoch,
        'data_idx': data_idx,
        'dir_suffix': dir_suffix
    }

def list_folders_in_dir(directory):
    folders = []
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        if os.path.isdir(path):
            folders.append(name)
    return folders


class TrainTask(Task):
    now_epoch_: int
    context_: TrainTaskContext
    config_: TrainTaskConfig
    recover_folder : str

    def __init__(self, config: TaskConfig, llm_name: str) -> None:
        super().__init__(config, llm_name)
        # init the default state
        # try recover the state

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
        if self.now_step_ > 1:
            self._pre_recover_context()


    def _pre_recover_state(self):
        parts = self.config_.adapter_.path_.split("/")
        first_part = parts[0]
        second_part = parts[1]
        folders = list_folders_in_dir(first_part)
        if folders:
            maxstep = 0
            for folder in folders:
                folderinfo = parse_output_dir(first_part + os.sep + folder, self.config_.adapter_.path_)
                # folderinfo: {'is_suffixed': , 'epoch': , 'data_idx': , 'dir_suffix': }
                if folderinfo["is_suffixed"] == False:
                    logging.info("train is finished")
                    exit()
                else:
                    if int(folderinfo["dir_suffix"]) > maxstep:
                        maxstep = int(folderinfo["dir_suffix"])
                        self.recover_folder = first_part + os.sep + folder
            recover_folder_info = parse_output_dir(self.recover_folder, self.config_.adapter_.path_)
            self.recover_folder = self.config_.adapter_.path_ + "_" + "_".join(
                [str(recover_folder_info["epoch"]),
                str(recover_folder_info["data_idx"]),
                str(recover_folder_info["dir_suffix"])]
            )
            self.now_epoch_ = int(recover_folder_info["epoch"]) + 1
            self.now_data_idx_ = int(recover_folder_info["data_idx"])
            self.now_step_ = int(recover_folder_info["dir_suffix"]) + 1
        else:
            self.now_epoch_ = 1
        
    def _pre_recover_context(self):
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
            output_dir = self.context_.path_ + "_" + "_".join(
                [str(self.now_epoch_), str(self.now_data_idx_), dir_suffix]
            )

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        weight_dict = self.context_.weight_dict()
        state_dict = self.context_.state_dict()
        # save to disk
        checkpoint = {
            "weight_dict" : weight_dict,
            "state_dict" : state_dict
        }
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
