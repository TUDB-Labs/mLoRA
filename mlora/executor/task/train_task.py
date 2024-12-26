import json
import logging
import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, override

import torch

import mlora.profiler
from mlora.config import TaskConfig, TrainTaskConfig
from mlora.executor.context import TrainTaskContext
from mlora.model.args import LinearInfo, Masks, MLoRADataConfig, Tokens
from mlora.model.tokenizer import Tokenizer

from .task import Task


def _get_context_state_from_folder_name(dir_name: str) -> Tuple[int, int, int]:
    split_group = dir_name.split("_")

    epoch = int(split_group[1])
    data_idx = int(split_group[2])
    step = int(split_group[3])

    return epoch, data_idx, step


class TrainTask(Task):
    now_epoch_: int

    context_: TrainTaskContext
    config_: TrainTaskConfig

    def __init__(self, config: TaskConfig, llm_name: str) -> None:
        super().__init__(config, llm_name)
        self.now_epoch_ = 1

    @override
    def is_done(self) -> bool:
        return self.now_epoch_ > self.config_.num_epochs_

    @override
    def prepare(self, linears_info: OrderedDict[str, LinearInfo], tokenizer: Tokenizer):
        self.tokenizer_ = tokenizer
        # prepare the context and the dataset
        # NOTE: how to recover the sort of dataset
        self._pre_context(linears_info)
        self._pre_recover_context()
        self._pre_dataset()

    def _get_recover_dir(self) -> str | None:
        if not os.path.isdir(self.context_.path_):
            return None

        def is_recover_dir(dir_name: str) -> bool:
            if "checkpoint" not in dir_name:
                return False

            if not os.path.isdir(os.path.join(self.context_.path_, dir_name)):
                return False

            return True

        recover_folders = list(filter(is_recover_dir, os.listdir(self.context_.path_)))

        if recover_folders is None or len(recover_folders) <= 0:
            return None

        max_step = -1
        max_epoch = -1
        to_recover_dir: str | None = None
        # Find the most suitable checkpoint as the recovery folder
        for folder in recover_folders:
            base_folder = os.path.basename(os.path.normpath(folder))
            step, epoch, data_idx = _get_context_state_from_folder_name(base_folder)
            # skip checkpoint that do not meet the condition
            if step is None or epoch > self.config_.num_epochs_:
                continue
            # Take maximum step, and take maximum epoch when steps are equal
            if step > max_step or (step == max_step and epoch > max_epoch):
                max_step = step
                max_epoch = epoch
                self.now_epoch_ = epoch
                self.now_data_idx_ = data_idx
                self.now_step_ = step
                # Set the recovery_folder name for restoring shuffle_data (if exist).
                self.recover_folder_ = folder
                to_recover_dir = os.path.join(self.context_.path_, folder)
        return to_recover_dir

    def _pre_recover_context(self):
        to_recover_dir = self._get_recover_dir()
        if to_recover_dir is None:
            return

        logging.info(
            f"Task {self.task_name()} have recover directory {to_recover_dir}"
            " need to recover."
        )
        self.checkpoint_ = True

        # get the optimizer read the file from now_epoch
        checkpoint = torch.load(to_recover_dir + os.sep + "checkpoint.bin")

        self.context_.recover_weight(checkpoint["weight_dict"])
        self.context_.recover_optimizer(checkpoint["state_dict"])
        # recompute the lr's epoch for recover
        lr_epoch = self.now_step_ // self.config_.accumulate_step_
        self.context_.recover_lr(lr_epoch)

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
            loss: torch.Tensor = self.context_.loss_fn_(loss_input, loss_target)

            logging.info(f"Adapter {self.context_.name_} loss: {loss}")

            mlora.profiler.metric_log(
                self.context_.path_ + "_loss", loss.item(), self.now_step_
            )

            return loss

        data_config = MLoRADataConfig(
            self.context_.name_,
            self.context_.type_,
            start_idx,
            end_idx,
            self._expand_batch_tokens,
            loss_fn,
            self.task_name(),
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

    def _save(
        self,
        is_checkpoint: bool = False,
        is_pipeline: Optional[int] = None,
        additional_info: Dict[str, str] = {},
    ):
        output_dir = self.context_.path_
        if is_pipeline is not None:
            output_dir = output_dir + os.sep + f"rank_{is_pipeline}"
        if is_checkpoint:
            checkpoint_folder = "checkpoint_" + "_".join(
                [
                    str(self.now_step_),
                    str(self.now_epoch_),
                    str(self.now_data_idx_),
                ]
            )
            output_dir = self.context_.path_ + os.sep + checkpoint_folder

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # save to disk, if save checkpoint, we need also save the state dict
        if is_checkpoint:
            torch.save(
                {
                    "weight_dict": self.context_.weight_dict(),
                    "state_dict": self.context_.state_dict(),
                },
                output_dir + os.sep + "checkpoint.bin",
            )
            # Save checkpoint for shuffle_data.
            self._save_data(output_dir)

        else:
            torch.save(
                self.context_.weight_dict(), output_dir + os.sep + "adapter_model.bin"
            )

        adapter_config: Dict[str, str] = {}
        adapter_config["base_model_name_or_path"] = self.llm_name_
        adapter_config = {**adapter_config, **additional_info}
        adapter_config = {**adapter_config, **self.config_.adapter_.export()}

        with open(output_dir + os.sep + "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=4)

    @override
    def done(self, is_pipeline: Optional[int] = None):
        self._save(is_checkpoint=False, is_pipeline=is_pipeline)
        # Delete the cache file.
        self._del_cache_file()
        # release the context
        del self.context_

    @override
    def terminate(self):
        del self.context_

    @override
    def step(self):
        stepd: bool = False
        need_checkpoint: bool = False

        if self.now_step_ % self.config_.accumulate_step_ == 0:
            stepd = True
            self.context_.step()

        if self.now_step_ % self.config_.save_step_ == 0:
            need_checkpoint = True

        self.now_step_ += 1
        self.now_data_idx_ += self.config_.mini_batch_size_

        if self.now_data_idx_ >= len(self.data_):
            self.now_epoch_ += 1
            self.now_data_idx_ = 0

        # to save the checkpoint, must ensure the order
        # beacuse we need recover the state
        if need_checkpoint:
            self._save(is_checkpoint=True)

        # task finish we also need to step
        if not stepd and self.now_epoch_ >= self.config_.num_epochs_:
            self.context_.step()

    @override
    def task_progress(self) -> int:
        total_step = len(self.data_) // self.config_.mini_batch_size_
        total_step = total_step * self.config_.num_epochs_
        return int((self.now_step_ / total_step) * 100)
