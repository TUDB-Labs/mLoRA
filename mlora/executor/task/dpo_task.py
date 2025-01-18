import copy
import logging
from typing import List, Optional, OrderedDict, Tuple, override

import torch
import torch.nn.functional as F

import mlora.profiler
from mlora.config import DPOTaskConfig
from mlora.executor.context import INFERENCECONTEXT_CLASS, TaskContext, TrainTaskContext
from mlora.model.args import LinearInfo, MLoRADataConfig, Tokens
from mlora.model.modules import AdapterModel
from mlora.model.tokenizer import Tokenizer

from .train_task import TrainTask


class DPOTask(TrainTask):
    config_: DPOTaskConfig
    context_: TrainTaskContext
    ref_context_: Optional[TaskContext]

    @override
    def prepare(self, linears_info: OrderedDict[str, LinearInfo], tokenizer: Tokenizer):
        self.tokenizer_ = tokenizer
        # prepare the dataset and context
        self._pre_dataset()
        self._pre_context(linears_info)
        self._pre_ref_context(linears_info)

        LOSS_CLASS = {"sigmoid": self.__dpo_loss_sigmoid, "ipo": self.__dpo_loss_ipo}

        self.context_.set_loss_fn(LOSS_CLASS[self.config_.loss_type_])

    def _pre_ref_context(self, linears_info: OrderedDict[str, LinearInfo]):
        if self.config_.reference_ is None:
            self.ref_context_ = None
            return

        ref_adapter_type = self.config_.reference_.type_
        self.ref_context_ = INFERENCECONTEXT_CLASS[ref_adapter_type](
            self.config_.reference_, linears_info
        )

    def __dpo_loss_sigmoid(self, logits: torch.Tensor) -> torch.Tensor:
        loss = (
            -F.logsigmoid(self.config_.beta_ * logits)
            * (1 - self.config_.label_smoothing_)
            - F.logsigmoid(-self.config_.beta_ * logits) * self.config_.label_smoothing_
        )
        return loss

    def __dpo_loss_ipo(self, logits: torch.Tensor) -> torch.Tensor:
        loss = (logits - 1 / (2 * self.config_.beta_)) ** 2
        return loss

    @override
    def adapter_model(self) -> List[AdapterModel]:
        if self.ref_context_ is None:
            return super().adapter_model()

        return [self.context_.adapter_model(), self.ref_context_.adapter_model()]

    @override
    def adapter_name(self) -> List[str]:
        if self.config_.reference_ is None:
            return super().adapter_name()

        return [self.config_.adapter_.name_, self.config_.reference_.name_]

    @override
    def switch_device(self, device: str):
        if self.ref_context_ is not None:
            self.ref_context_.switch_device(device)

        self.context_.switch_device(device)

    @override
    def data(self, start_idx: int) -> Tuple[List[Tokens], List[MLoRADataConfig]]:
        logging.info(
            f"Task - {self.context_.name_} "
            f"epoch: {self.now_epoch_}/{self.config_.num_epochs_}"
            f" iteration: {self.now_data_idx_}/{len(self.data_)} step: {self.now_step_}"
        )
        data_idx_s = self.now_data_idx_
        data_idx_e = self.now_data_idx_ + self.config_.mini_batch_size_

        # 0...mid is chosen data
        # mid.end is reject data
        batch_str = self.prompter_.generate_prompt(self.data_[data_idx_s:data_idx_e])

        assert len(batch_str) % 2 == 0

        ret_tokens = []
        # for refrerence
        ref_model_token = list(
            map(
                lambda raw_str: self.tokenizer_.encode(
                    raw_str, bos=True, eos=True, cutoff_len=self.config_.cutoff_len_
                ),
                batch_str,
            )
        )
        policy_model_token = copy.deepcopy(ref_model_token)

        ret_tokens.extend(ref_model_token)
        ret_tokens.extend(policy_model_token)

        # include reference and policy models' chosen and reject data
        assert len(ret_tokens) % 4 == 0

        ref_start_idx = start_idx
        ref_end_idx = ref_start_idx + len(ref_model_token)

        policy_start_idx = ref_end_idx
        policy_end_idx = policy_start_idx + len(policy_model_token)

        def loss_fn(
            input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
        ) -> torch.Tensor:
            mask = ~mask[ref_start_idx:policy_end_idx, 1:]

            logits = input[ref_start_idx:policy_end_idx, :-1, :].log_softmax(-1)
            labels = target[ref_start_idx:policy_end_idx, 1:].to(input.device)
            mask = mask.long().to(input.device)

            per_token_logps = torch.gather(
                logits, dim=2, index=labels.unsqueeze(2)
            ).squeeze(2)

            logps = (per_token_logps * mask).sum(-1)

            data_len = policy_end_idx - ref_start_idx
            assert data_len % 4 == 0
            data_len = data_len // 4

            ref_chosen_logps, ref_reject_logps, pi_chosen_logps, pi_reject_logps = [
                logps[i * data_len : (i + 1) * data_len] for i in range(4)
            ]

            pi = pi_chosen_logps - pi_reject_logps
            ri = ref_chosen_logps - ref_reject_logps

            chosen_reward = (pi_chosen_logps - ref_chosen_logps) * self.config_.beta_
            reject_reward = (pi_reject_logps - ref_reject_logps) * self.config_.beta_

            logits = pi - ri

            loss: torch.Tensor = self.context_.loss_fn_(logits)
            loss = loss.mean()

            mlora.profiler.metric_log(
                self.context_.path_ + "_loss", loss.item(), self.now_step_
            )
            mlora.profiler.metric_log(
                self.context_.path_ + "_chosen_reward",
                chosen_reward.mean().item(),
                self.now_step_,
            )
            mlora.profiler.metric_log(
                self.context_.path_ + "_rejected_reward",
                reject_reward.mean().item(),
                self.now_step_,
            )

            logging.info(
                f"Task - {self.context_.name_} loss: {loss}, "
                f"chosen_rewards: {chosen_reward.mean()}, "
                f"rejected_rewards: {reject_reward.mean()}"
            )

            return loss

        ref_model_name = ""
        ref_model_type = ""

        if self.ref_context_ is not None:
            ref_model_name = self.ref_context_.name_
            ref_model_type = self.ref_context_.type_

        ref_model_config = MLoRADataConfig(
            ref_model_name,
            ref_model_type,
            ref_start_idx,
            ref_end_idx,
            self._expand_batch_tokens,
            lambda *_: None,
            self.task_name(),
        )

        policy_model_config = MLoRADataConfig(
            self.context_.name_,
            self.context_.type_,
            policy_start_idx,
            policy_end_idx,
            self._expand_batch_tokens,
            loss_fn,
            self.task_name(),
        )

        return ret_tokens, [ref_model_config, policy_model_config]

    @override
    def done(self, is_pipeline: Optional[int] = None):
        self._save(is_pipeline=is_pipeline)
        # release the context
        del self.context_
        if self.ref_context_ is not None:
            del self.ref_context_

    @override
    def terminate(self):
        # release the context
        del self.context_
        if self.ref_context_ is not None:
            del self.ref_context_
