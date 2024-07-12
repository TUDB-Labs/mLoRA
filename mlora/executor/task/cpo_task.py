import logging
from collections import OrderedDict
from typing import List, Tuple, override

import torch
import torch.nn.functional as F

import mlora.profiler
from mlora.config import CPOTaskConfig
from mlora.executor.context import TrainLoRAContext
from mlora.model.args import LinearInfo, MLoRADataConfig, Tokens
from mlora.model.tokenizer import Tokenizer

from .train_task import TrainTask


class CPOTask(TrainTask):
    context_: TrainLoRAContext
    config_: CPOTaskConfig

    @override
    def prepare(self, linears_info: OrderedDict[str, LinearInfo], tokenizer: Tokenizer):
        self.tokenizer_ = tokenizer
        # prepare the dataset and context
        self._pre_dataset()
        self._pre_context(linears_info)

        LOSS_CLASS = {
            "sigmoid": self.__cpo_loss_sigmoid,
            "hinge": self.__cpo_loss_hinge,
        }

        self.context_.set_loss_fn(LOSS_CLASS[self.config_.loss_type_])

    def __cpo_loss_sigmoid(self, logits: torch.Tensor) -> torch.Tensor:
        return -F.logsigmoid(self.config_.beta_ * logits)

    def __cpo_loss_hinge(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.relu(1 - self.config_.beta_ * logits)

    @override
    def data(self, start_idx: int) -> Tuple[List[Tokens], List[MLoRADataConfig]]:
        logging.info(
            f"Adapter {self.context_.name_} epoch: {
                self.now_epoch_}/{self.config_.num_epochs_}"
            f" iteration: {self.now_data_idx_}/{len(self.data_)} step: {self.now_step_}"
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
            input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
        ) -> torch.Tensor:
            mask = ~mask[start_idx:end_idx:, 1:]

            data_len = end_idx - start_idx
            assert data_len % 2 == 0
            data_len = data_len // 2

            batch_input = input[start_idx:end_idx, :-1, :].contiguous()
            batch_label = target[start_idx:end_idx, 1:].contiguous().to(input.device)
            mask = mask.long().to(input.device)

            # step1. calc the chose loss
            vacab_size = input.shape[-1]
            chose_input = batch_input[:data_len].view(-1, vacab_size)
            chose_label = batch_label[:data_len].view(-1)
            loss_chosen: torch.Tensor = F.cross_entropy(chose_input, chose_label)

            # step2. calc the prefer loss
            logits = batch_input.log_softmax(-1)
            per_token_logps = torch.gather(
                logits, dim=2, index=batch_label.unsqueeze(2)
            ).squeeze(2)

            logps = (per_token_logps * mask).sum(-1)

            chosen_logps, reject_logps = logps[:data_len], logps[data_len:]

            logits = chosen_logps - reject_logps
            loss_prefer = self.context_.loss_fn_(logits)

            loss = loss_prefer.mean() + loss_chosen

            mlora.profiler.metric_log(
                self.context_.path_ + "_loss", loss.item(), self.now_step_
            )
            mlora.profiler.metric_log(
                self.context_.path_ + "_loss_prefer",
                loss_prefer.mean().item(),
                self.now_step_,
            )
            logging.info(f"Adapter {self.context_.name_} loss: {loss}")
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
