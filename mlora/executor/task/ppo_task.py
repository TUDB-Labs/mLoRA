import copy
import json
import logging
import os
from enum import Enum
from typing import Callable, Dict, List, Optional, OrderedDict, Tuple, override

import torch
from datasets import load_dataset
from torch.distributions import Categorical
from tqdm import tqdm

from mlora.config import PPOTaskConfig
from mlora.executor.context import (
    INFERENCECONTEXT_CLASS,
    TRAINCONTEXT_CLASS,
    TaskContext,
    TrainTaskContext,
)
from mlora.model.args import LinearInfo, MLoRADataConfig, Tokens
from mlora.model.modules import AdapterModel
from mlora.model.tokenizer import Tokenizer
from mlora.prompter import PrompterFactory

from .train_task import TrainTask


class PPOTrainStage(Enum):
    REWARD_MODEL_TRAINING = 0
    INIT = 1
    DECISION = 2
    UPDATE = 3
    ITERATION = 4


class PPOTask(TrainTask):

    reward_linear_tensor: torch.Tensor = torch.zeros((1, 1), requires_grad=False)
    critic_linear_tensor: torch.Tensor = torch.zeros((1, 1), requires_grad=False)

    reward_context_: TrainTaskContext
    critic_context_: TrainTaskContext
    actor_context_: TrainTaskContext
    ref_context_: Optional[TaskContext]
    config_: PPOTaskConfig
    generate_index: int
    now_K_epochs: int
    now_optim_iter_num: int
    advantage: torch.Tensor
    td_target: torch.Tensor
    policy_tokens: list[list[int]]
    stage: PPOTrainStage

    def __init__(self, config: PPOTaskConfig, llm_name: str) -> None:
        super().__init__(config, llm_name)
        self.policy_tokens = []
        self.stage = PPOTrainStage.REWARD_MODEL_TRAINING
        self.now_epoch_ = 0
        self.now_K_epochs = 0
        self.now_optim_iter_num = 0
        self.eps = 1e-6
        self.generate_index = 0
        self.advantage = torch.zeros(1)
        self.td_target = torch.zeros(1)
        self.perm = torch.zeros(1)

    def __normalize(self, loss: torch.Tensor) -> torch.Tensor:
        loss_mean = loss.mean()
        loss_std = loss.std()
        normalized_loss = (loss - loss_mean) / (loss_std + self.eps)
        return normalized_loss

    def reward_func(self, reward_t: torch.Tensor) -> torch.Tensor:
        dim = reward_t.shape[-1]
        device = self.td_target.device
        if PPOTask.reward_linear_tensor.shape[0] != dim:
            PPOTask.reward_linear_tensor = torch.randn(
                (dim, 1), requires_grad=False, device=device
            )
        return self.__normalize(reward_t @ PPOTask.reward_linear_tensor)

    def critic_func(self, critic_t: torch.Tensor) -> torch.Tensor:
        dim = critic_t.shape[-1]
        device = self.td_target.device
        if PPOTask.critic_linear_tensor.shape[0] != dim:
            PPOTask.critic_linear_tensor = torch.randn(
                (dim, 1), requires_grad=False, device=device
            )
        return self.__normalize(critic_t @ PPOTask.critic_linear_tensor)

    def prepare(self, linears_info: OrderedDict[str, LinearInfo], tokenizer: Tokenizer):
        self.tokenizer_ = tokenizer

        # prepare the context and the dataset
        # NOTE: how to recover the sort of dataset
        self._pre_dataset()
        self.ppo_pre_context(linears_info)

        LOSS_CLASS = {
            "mse": PPOTask.ppo_mse,
            "adv_loss": PPOTask.ppo_adv_loss,
            "reward_loss": PPOTask.ppo_reward_loss,
        }
        self.critic_context_.set_loss_fn(
            LOSS_CLASS[self.config_.critic_loss_type_]  # type: ignore
        )
        self.actor_context_.set_loss_fn(
            LOSS_CLASS[self.config_.actor_loss_type_]  # type: ignore
        )
        self.reward_context_.set_loss_fn(
            LOSS_CLASS[self.config_.reward_loss_type_]  # type: ignore
        )

    def _pre_dataset(self):
        preprocess_func: Dict[str, Callable] = {
            "default": lambda data: data,
            "shuffle": lambda data: self._shuffle_data(data),
            "sort": lambda data: data.sort(),
        }

        if self.config_.dataset_ is None:
            logging.info(
                "Task dataset is empty, maybe in pipeline we do not load dataset."
            )
            return

        self.prompter_ = PrompterFactory.create(self.config_.dataset_)

        logging.info(f"Task load data from {self.config_.dataset_.data_path_}")
        data = load_dataset(
            "json", data_files={"data_points": self.config_.dataset_.data_path_}
        )

        preprocess_type = self.config_.dataset_.preprocess_
        if preprocess_type not in preprocess_func:
            raise NotImplementedError

        # Process data according to the data preprocess_type.
        data = preprocess_func[preprocess_type](data)
        logging.info(
            f"Adapters {', '.join(self.adapter_name())} "
            f"data size: {len(data['data_points'])}"
        )

        for _, data_point in tqdm(enumerate(data["data_points"])):
            self.data_.append(data_point)

    def _pre_ref_context(self, linears_info: OrderedDict[str, LinearInfo]):
        if self.config_.reference_ is None:
            self.ref_context_ = None
            return

        ref_adapter_type = self.config_.reference_.type_
        self.ref_context_ = INFERENCECONTEXT_CLASS[ref_adapter_type](
            self.config_.reference_, linears_info
        )

    def ppo_pre_context(self, linears_info: OrderedDict[str, LinearInfo]):
        reward_adapter_type_ = self.config_.reward_adapter_.type_
        critic_adapter_type_ = self.config_.critic_adapter_.type_
        actor_adapter_type_ = self.config_.actor_adapter_.type_

        assert reward_adapter_type_ in TRAINCONTEXT_CLASS
        assert critic_adapter_type_ in TRAINCONTEXT_CLASS
        assert actor_adapter_type_ in TRAINCONTEXT_CLASS

        self.reward_context_ = TRAINCONTEXT_CLASS[reward_adapter_type_](
            self.config_.reward_adapter_, linears_info
        )
        self.critic_context_ = TRAINCONTEXT_CLASS[critic_adapter_type_](
            self.config_.critic_adapter_, linears_info
        )
        self.actor_context_ = TRAINCONTEXT_CLASS[actor_adapter_type_](
            self.config_.actor_adapter_, linears_info
        )
        self._pre_ref_context(linears_info)

    @staticmethod
    def ppo_mse(data: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return (data - label).pow(2).mean()

    @staticmethod
    def ppo_adv_loss(
        prob: torch.Tensor,
        old_prob: torch.Tensor,
        advantage: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:

        clip_rate_ = 0.2
        prob_a = prob.gather(-1, action).squeeze(-1)
        old_prob_a = old_prob.gather(-1, action).squeeze(-1)
        ratio = torch.exp(torch.log(prob_a) - torch.log(old_prob_a))

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - clip_rate_, 1 + clip_rate_) * advantage
        loss1 = -torch.min(surr1, surr2)
        loss1 = loss1.view(-1).mean()

        return loss1

    @staticmethod
    def ppo_reward_loss(
        reward_chosen: torch.Tensor, reward_reject: torch.Tensor
    ) -> torch.Tensor:
        effective_reward = torch.maximum(
            reward_chosen - reward_reject, torch.tensor(0.0)
        )
        loss = -torch.mean(torch.log(torch.sigmoid(effective_reward)))
        return loss

    @override
    def adapter_model(self) -> List[AdapterModel]:

        adapter_model_sequence = [
            self.reward_context_.adapter_model(),
            self.actor_context_.adapter_model(),
            self.critic_context_.adapter_model(),
        ]
        if self.ref_context_ is not None:
            adapter_model_sequence.append(self.ref_context_.adapter_model())

        return adapter_model_sequence

    @override
    def adapter_name(self) -> list[str]:

        adapter_name_sequence = [
            self.config_.reward_adapter_.name_,
            self.config_.actor_adapter_.name_,
            self.config_.critic_adapter_.name_,
        ]
        if self.config_.reference_ is not None:
            adapter_name_sequence.append(self.config_.reference_.name_)

        return adapter_name_sequence

    @override
    def switch_device(self, device: str):
        if self.ref_context_ is not None:
            self.ref_context_.switch_device(device)
        self.critic_context_.switch_device(device)
        self.actor_context_.switch_device(device)
        self.reward_context_.switch_device(device)
        self.advantage = self.advantage.to(device)
        self.td_target = self.td_target.to(device)
        self.perm = self.perm.to(device)

        PPOTask.reward_linear_tensor = PPOTask.reward_linear_tensor.to(device)
        PPOTask.critic_linear_tensor = PPOTask.critic_linear_tensor.to(device)

    def stage_reward_training(self, start_idx: int):

        data_idx_s = self.now_data_idx_
        data_idx_e = self.now_data_idx_ + self.config_.mini_batch_size_
        # get the train raw string
        batch_strr = self.prompter_.generate_prompt(self.data_[data_idx_s:data_idx_e])
        reward_tokens = list(
            map(
                lambda raw_str: self.tokenizer_.encode(
                    raw_str, bos=True, eos=True, cutoff_len=self.config_.cutoff_len_
                ),
                batch_strr,
            )
        )

        reward_batch = int(len(reward_tokens) / 3)
        reward_tokens = reward_tokens[reward_batch:]
        reward_start_idx = start_idx
        reward_end_idx = reward_start_idx + len(reward_tokens)

        def loss_fn(input: torch.Tensor, _: torch.Tensor, mask: torch.Tensor):

            mask = ~mask[reward_start_idx:reward_end_idx]
            reward = self.reward_func(input)
            reward = reward.squeeze(dim=-1)
            reward_batch = int(len(reward) / 2)
            mask = mask.to(self.td_target.device)
            reward_chosen = reward[:reward_batch] * mask[:reward_batch]
            reward_reject = reward[reward_batch:] * mask[reward_batch:]
            loss = self.reward_context_.loss_fn_(reward_chosen, reward_reject)

            logging.info(f"Adapter {self.reward_context_.name_} loss: {loss} ")

            return loss

        reward_data_config = MLoRADataConfig(
            self.reward_context_.name_,
            self.reward_context_.type_,
            reward_start_idx,
            reward_end_idx,
            self._expand_batch_tokens,
            loss_fn,
            self.task_name(),
        )

        return reward_tokens, [reward_data_config]

    def stage_init(self, start_idx: int):
        logging.info("reward_model's ready")

        data_idx_s = self.now_data_idx_
        data_idx_e = self.now_data_idx_ + self.config_.mini_batch_size_

        # get the train raw string
        batch_str = self.prompter_.generate_prompt(self.data_[data_idx_s:data_idx_e])
        actor_tokens = list(
            map(
                lambda raw_str: self.tokenizer_.encode(
                    raw_str, bos=True, eos=True, cutoff_len=self.config_.cutoff_len_
                ),
                batch_str,
            )
        )

        actor_tokens_batch = int(len(actor_tokens) / 3)
        actor_tokens = actor_tokens[:actor_tokens_batch]

        self.policy_tokens = actor_tokens

        self.stage = PPOTrainStage.DECISION

        return self.stage_decision(start_idx)

    def stage_decision(
        self,
        start_idx: int,
    ):
        batch_num = int(len(self.policy_tokens))
        actor_len = int(len(self.policy_tokens[0]))
        generate_text_len = self.config_.generate_num_
        actor_start_idx = start_idx
        actor_end_idx = actor_start_idx + batch_num
        actor_tokens = copy.deepcopy(self.policy_tokens)

        def loss_fn(
            input: torch.Tensor,
            _: torch.Tensor,
            __: torch.Tensor,
            deterministic: bool = False,
        ) -> Optional[torch.Tensor]:

            if deterministic:
                a = torch.argmax(
                    input[actor_start_idx:actor_end_idx, actor_len - 1], dim=-1
                )
            else:
                input_ = torch.softmax(
                    input[actor_start_idx:actor_end_idx, actor_len - 1].view(
                        -1, input[actor_start_idx:actor_end_idx].shape[-1]
                    ),
                    dim=-1,
                )
                m = Categorical(input_)
                a = m.sample()
                a = a.view(batch_num, -1)

            for token_list, action in zip(self.policy_tokens, a):
                token_list.append(int(action.item()))
            self.generate_index += 1

            if self.generate_index == generate_text_len:
                self.stage = PPOTrainStage.UPDATE

            return None

        actor_data_config = MLoRADataConfig(
            self.actor_context_.name_,
            self.actor_context_.type_,
            actor_start_idx,
            actor_end_idx,
            self._expand_batch_tokens,
            loss_fn,
            self.task_name(),
        )

        return actor_tokens, [actor_data_config]

    def stage_update(self, start_idx: int):

        batch_num = int(len(self.policy_tokens))
        ref_len = int(len(self.policy_tokens[0]))
        actor_len = int(len(self.policy_tokens[0]))
        critic_len = actor_len
        generate_text_len = self.config_.generate_num_
        reward_tokens = copy.deepcopy(self.policy_tokens)
        ref_tokens = copy.deepcopy(self.policy_tokens)
        actor_tokens = copy.deepcopy(self.policy_tokens)
        critic_tokens = copy.deepcopy(self.policy_tokens)
        p_tokens: List[List[int]] = (
            reward_tokens + ref_tokens + actor_tokens + critic_tokens
        )

        reward_start_idx = start_idx
        reward_end_idx = reward_start_idx + batch_num
        ref_start_idx = reward_end_idx
        ref_end_idx = ref_start_idx + batch_num
        actor_start_idx = ref_end_idx
        actor_end_idx = actor_start_idx + batch_num
        critic_start_idx = actor_end_idx
        critic_end_idx = critic_start_idx + batch_num

        def loss_fn(
            input: torch.Tensor, _: torch.Tensor, __: torch.Tensor
        ) -> Optional[torch.Tensor]:

            # Dividing a long trajectory into shorter trajectories for updating
            assert generate_text_len % self.config_.optim_num_ == 0
            data_len = int(generate_text_len / self.config_.optim_num_)

            actor_slice = input[
                actor_start_idx:actor_end_idx,
                actor_len - generate_text_len - 1 : actor_len - 1,
            ]
            ref_slice = input[
                ref_start_idx:ref_end_idx, ref_len - generate_text_len - 1 : ref_len - 1
            ]
            p = actor_slice.softmax(dim=-1)
            log_p = actor_slice.log_softmax(dim=-1)
            log_ref_p = ref_slice.log_softmax(dim=-1)
            action = torch.tensor(self.policy_tokens, device=self.advantage.device)[
                :, actor_len - generate_text_len :
            ].unsqueeze(dim=-1)
            log_prob = log_p.gather(-1, action).squeeze(-1)
            ref_log_prob = log_ref_p.gather(-1, action).squeeze(-1)
            r = -self.config_.kl_coefficient_ * (log_prob - ref_log_prob)
            r[:, -1] += (
                self.reward_func(input[reward_start_idx:reward_end_idx, actor_len - 1])
            ).squeeze(dim=-1)

            v = (
                self.critic_func(
                    input[
                        critic_start_idx:critic_end_idx,
                        critic_len - generate_text_len - 1 : critic_len,
                    ]
                )
            ).squeeze(dim=-1)

            v_ = v.clone().detach()
            v_[:, -1] = 0

            # For multiple updates,we need to record the initial advantage
            if self.now_K_epochs == 0 and self.now_optim_iter_num == 0:
                deltas = torch.zeros_like(v_)
                for j in range(1, len(deltas[0])):
                    deltas[:, j - 1] = (
                        r[:, j - 1] + self.config_.gamma_ * v_[:, j] - v_[:, j - 1]
                    )

                advantage = torch.zeros_like(v_)

                for j in range(len(advantage[0]) - 2, -1, -1):
                    advantage[:, j] = (
                        deltas[:, j]
                        + self.config_.gamma_
                        * self.config_.lamdb_
                        * advantage[:, j + 1]
                    )
                advantage = self.__normalize(advantage)

                advantage = torch.flip(advantage, [-1])
                advantage = advantage[:, 0:-1]
                v_ = v_[:, 0:-1]
                td_target = advantage + v_
                self.advantage = advantage
                self.td_target = td_target
                self.old_p = p

            if self.now_optim_iter_num == 0:
                self.perm = torch.randperm(len(self.advantage[0]))

            advantage_ = self.advantage[:, self.perm].clone().detach()
            td_target_ = self.td_target[:, self.perm].clone().detach()
            old_p = self.old_p[:, self.perm].clone().detach()
            p = p[:, self.perm]
            v_ = v_[:, self.perm]
            action = action.to(self.advantage.device)
            action = action[:, self.perm]

            index = [
                i
                for i in range(
                    self.now_optim_iter_num * data_len,
                    min(
                        (self.now_optim_iter_num + 1) * data_len, len(self.advantage[0])
                    ),
                )
            ]
            loss1 = self.critic_context_.loss_fn_(
                v_[:, index].view(-1), td_target_[:, index].view(-1)
            )
            loss2 = self.actor_context_.loss_fn_(
                p[:, index], old_p[:, index], advantage_[:, index], action[:, index]
            )
            loss = loss1 + loss2

            self.now_optim_iter_num += 1
            if self.now_optim_iter_num == self.config_.optim_num_:
                self.now_K_epochs += 1
                self.now_optim_iter_num = 0

            if self.now_K_epochs == self.config_.K_epochs_:
                self.now_K_epochs = 0
                self.stage = PPOTrainStage.ITERATION

            logging.info(
                f"Adapter {self.critic_context_.name_} loss: {loss1} "
                f"Adapter {self.actor_context_.name_} loss: {loss2} "
            )

            return loss

        ref_model_name = ""
        ref_model_type = ""
        if self.ref_context_ is not None:
            ref_model_name = self.ref_context_.name_
            ref_model_type = self.ref_context_.type_

        reward_data_config = MLoRADataConfig(
            self.reward_context_.name_,
            self.reward_context_.type_,
            reward_start_idx,
            reward_end_idx,
            self._expand_batch_tokens,
            lambda *_: None,
            self.task_name(),
        )
        ref_data_config = MLoRADataConfig(
            ref_model_name,
            ref_model_type,
            ref_start_idx,
            ref_end_idx,
            self._expand_batch_tokens,
            lambda *_: None,
            self.task_name(),
        )
        actor_data_config = MLoRADataConfig(
            self.actor_context_.name_,
            self.actor_context_.type_,
            actor_start_idx,
            actor_end_idx,
            self._expand_batch_tokens,
            lambda *_: None,
            self.task_name(),
        )
        critic_data_config = MLoRADataConfig(
            self.critic_context_.name_,
            self.critic_context_.type_,
            critic_start_idx,
            critic_end_idx,
            self._expand_batch_tokens,
            loss_fn,
            self.task_name(),
        )

        return p_tokens, [
            reward_data_config,
            ref_data_config,
            actor_data_config,
            critic_data_config,
        ]

    @override
    def data(self, start_idx: int) -> Tuple[List[Tokens], List[MLoRADataConfig]]:

        stages = [
            self.stage_reward_training,
            self.stage_init,
            self.stage_decision,
            self.stage_update,
        ]

        data, data_config = stages[self.stage.value](start_idx)

        logging.info(
            f"Task - {self.reward_context_.name_}, "
            f"{self.actor_context_.name_}, {self.critic_context_.name_} "
            f"epoch: {self.now_epoch_}/{self.config_.num_epochs_} "
            f"iteration: {self.now_data_idx_}/{len(self.data_)} step: {self.now_step_} "
            f"state: {self.stage} "
            f"generate_index: {self.generate_index} "
            f"now_K_epoch: {self.now_K_epochs} "
            f"now_optim_num: {self.now_optim_iter_num}"
        )

        return data, data_config

    @override
    def step(self):
        if self.stage in [PPOTrainStage.INIT, PPOTrainStage.DECISION]:
            return

        stepd, need_checkpoint = False, False
        is_reward_training = self.stage == PPOTrainStage.REWARD_MODEL_TRAINING

        # Perform training step if necessary
        if self.now_step_ % self.config_.accumulate_step_ == 0:
            stepd = True
            self._perform_training_step(is_reward_training)

        # Check if checkpoint is needed
        if self.now_step_ % self.config_.save_step_ == 0:
            need_checkpoint = True

        # Increment the step counter
        self.now_step_ += 1

        # Handle specific state changes
        self._update_data_idx_and_state(is_reward_training)

        # Check if data is exhausted, increment epoch
        if self.now_data_idx_ >= len(self.data_):
            self.now_epoch_ += 1
            self.now_data_idx_ = 0

        # Save checkpoint if needed
        if need_checkpoint:
            self._save(is_checkpoint=True)

        # Final step if training has finished
        if not stepd and self.now_epoch_ == self.config_.num_epochs_:
            self._perform_training_step(is_reward_training)

        # Reset state after completing reward training
        if is_reward_training and self.now_epoch_ == self.config_.num_epochs_:
            self._reset_training_state()

    def _perform_training_step(self, is_reward_training):
        if is_reward_training:
            self.reward_context_.step()
        else:
            self.critic_context_.step()
            self.actor_context_.step()

    def _update_data_idx_and_state(self, is_reward_training):
        if self.stage == PPOTrainStage.REWARD_MODEL_TRAINING:
            self.now_data_idx_ += self.config_.mini_batch_size_
        elif self.stage == PPOTrainStage.ITERATION:
            self.now_data_idx_ += self.config_.mini_batch_size_
            self.stage = PPOTrainStage.INIT
            self.generate_index = 0

    def _reset_training_state(self):
        self.stage = PPOTrainStage.INIT
        self.now_epoch_ = 0

    def __save(
        self,
        context_: TrainTaskContext,
        is_checkpoint: bool = False,
        is_pipeline: Optional[int] = None,
        additional_info: Dict[str, str] = {},
    ):
        output_dir = context_.path_
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
            output_dir = context_.path_ + os.sep + checkpoint_folder

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # save to disk, if save checkpoint, we need also save the state dict
        if is_checkpoint:
            torch.save(
                {
                    "weight_dict": context_.weight_dict(),
                    "state_dict": context_.state_dict(),
                },
                output_dir + os.sep + "checkpoint.bin",
            )
            # Save checkpoint for shuffle_data.
            self._save_data(output_dir)

        else:
            torch.save(
                context_.weight_dict(), output_dir + os.sep + "adapter_model.bin"
            )

        adapter_config: Dict[str, str] = {}
        tmp_dict: Dict[str, str] = {}
        if "reward" in context_.name_:
            tmp_dict = self.config_.reward_adapter_.export()
        elif "actor" in context_.name_:
            tmp_dict = self.config_.actor_adapter_.export()
        else:
            tmp_dict = self.config_.critic_adapter_.export()
        adapter_config["base_model_name_or_path"] = self.llm_name_
        adapter_config = {**adapter_config, **additional_info}

        adapter_config = {**adapter_config, **tmp_dict}

        with open(output_dir + os.sep + "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=4)

    def _save(
        self,
        is_checkpoint: bool = False,
        is_pipeline: Optional[int] = None,
        additional_info: Dict[str, str] = {},
    ):
        self.__save(self.actor_context_, is_checkpoint, is_pipeline, additional_info)
        self.__save(self.critic_context_, is_checkpoint, is_pipeline, additional_info)
        self.__save(self.reward_context_, is_checkpoint, is_pipeline, additional_info)

    @override
    def done(self, is_pipeline: Optional[int] = None):
        self._save(is_checkpoint=False, is_pipeline=is_pipeline)
        # Delete the cache file.
        self._del_cache_file()
        # release the context
        del self.critic_context_
        del self.actor_context_
        del self.reward_context_
        if self.ref_context_ is not None:
            del self.ref_context_

    @override
    def terminate(self):
        del self.critic_context_
        del self.actor_context_
        del self.reward_context_
        if self.ref_context_ is not None:
            del self.ref_context_
