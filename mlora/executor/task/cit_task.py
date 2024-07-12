import logging
from typing import List, Tuple, override

import torch
import torch.nn.functional as F

import mlora.profiler
from mlora.config import CITTaskConfig
from mlora.executor.context import TrainLoRAContext
from mlora.model.args import MLoRADataConfig, Tokens

from .train_task import TrainTask


class CITTask(TrainTask):
    context_: TrainLoRAContext
    config_: CITTaskConfig
    pooling_method_: str

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
            # step1. calc the contrastive loss
            original_entire_hidden_states = batch_input[:data_len]
            paraphrased_entire_hidden_states = batch_input[data_len:]
            if self.config_.contrastive_pooling_method_ == "last":
                original_hidden_states = original_entire_hidden_states[:, -1, :]
                paraphrased_hidden_states = paraphrased_entire_hidden_states[:, -1, :]

            batch_size = original_hidden_states.size(0)
            labels = torch.arange(batch_size).to(original_hidden_states.device)
            original_to_paraphrased_sim = F.cosine_similarity(
                original_hidden_states.unsqueeze(1),
                paraphrased_hidden_states.unsqueeze(0),
                dim=2,
            )
            original_to_paraphrased_sim = (
                original_to_paraphrased_sim / self.config_.temperature_
            )
            paraphrased_to_original_sim = original_to_paraphrased_sim.T
            ori_to_para_loss = F.cross_entropy(original_to_paraphrased_sim, labels)
            para_to_ori_loss = F.cross_entropy(paraphrased_to_original_sim, labels)
            contrastive_loss = (ori_to_para_loss + para_to_ori_loss) / 2

            # step2. calc the generation loss
            vacab_size = input.shape[-1]
            batch_input = batch_input.view(-1, vacab_size)
            batch_label = batch_label.view(-1)
            loss_generation: torch.Tensor = F.cross_entropy(batch_input, batch_label)
            contras_raito = min(
                self.config_.lambda_,
                loss_generation.detach().item() / contrastive_loss.detach().item(),
            )
            loss = contrastive_loss.mean() * contras_raito + loss_generation

            logging.info(f"Adapter {self.context_.name_} loss: {loss}")
            mlora.profiler.metric_log(
                self.context_.path_ + "_loss", loss.item(), self.now_step_
            )
            mlora.profiler.metric_log(
                self.context_.path_ + "_loss_contrastive",
                contrastive_loss.item(),
                self.now_step_,
            )
            mlora.profiler.metric_log(
                self.context_.path_ + "_loss_generation",
                loss_generation.item(),
                self.now_step_,
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
