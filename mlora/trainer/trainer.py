from mlora.model.model import LLMModel
from mlora.dispatcher.dispatcher import Dispatcher
from mlora.config import LoraConfig, MLoRAConfig, OptimConfig, TrainerConfig
from mlora.model.modelargs import MultiLoraBatchData

import os
import json
import torch

from typing import Dict, List


class TrainerContext:
    name_or_path_: str = ""
    adapter_name_: str = ""
    loss_fn_: torch.nn.Module = None
    accumulation_step_: int = -1
    step_cnt_: int = -1
    optimizer_: torch.optim.Optimizer = None
    save_step_: int = 1000

    def __init__(self,
                 lora_config: LoraConfig,
                 tainer_config: TrainerConfig,
                 trainable_params: torch.Tensor):
        self.adapter_name_ = lora_config.adapter_name_

        self.loss_fn_ = torch.nn.CrossEntropyLoss()

        self.batch_size_ = lora_config.batch_size_
        self.micro_batch_size_ = lora_config.micro_batch_size_
        if self.batch_size_ < self.micro_batch_size_ or self.batch_size_ % self.micro_batch_size_ != 0:
            raise f"error batch_size {self.batch_size_} and micro batch size {self.micro_batch_size_}"
        self.accumulation_step_ = self.batch_size_ / self.micro_batch_size_
        self.step_cnt_ = 0
        self.save_step_ = tainer_config.save_step_

        self.setup_optimizer(lora_config.optim_config_, trainable_params)

        assert self.optimizer_ is not None

        # config
        self.alpha_ = lora_config.lora_alpha_
        self.dropout_ = lora_config.lora_dropout_
        self.r_ = lora_config.r_
        self.target_modules_ = lora_config.target_

    def setup_optimizer(self,
                        config: OptimConfig,
                        trainable_params: torch.Tensor):
        if config.optim_ == "adamw":
            from mlora.config import AdamWOptimConfig
            assert isinstance(config, AdamWOptimConfig)
            self.optimizer_ = torch.optim.AdamW(
                trainable_params, lr=config.lr_)
        elif config.optim_ == "sgd":
            from mlora.config import SGDOptimConfig
            assert isinstance(config, SGDOptimConfig)
            self.optimizer_ = torch.optim.SGD(
                trainable_params, lr=config.lr_, momentum=config.momentum_)
        else:
            raise f"unkown optimizer {self.optim_name_}"

    def step(self):
        self.step_cnt_ += 1
        if self.step_cnt_ % self.accumulation_step_ == 0:
            self.optimizer_.step()
            self.optimizer_.zero_grad()

    def finish(self):
        self.optimizer_.step()
        self.optimizer_.zero_grad()

    def export_config(self) -> Dict[str, str]:
        return {
            "base_model_name_or_path": self.name_or_path_,
            "lora_alpha": self.alpha_,
            "lora_dropout": self.dropout_,
            "r": self.r_,
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "bias": "none",
            "target_modules": [key for key in self.target_modules_ if self.target_modules_[key]]
        }

    def is_save_step(self) -> bool:
        return self.step_cnt_ % self.save_step_ == 0


class MutiTrainerContext:
    trainer_context_: Dict[str, TrainerContext] = {}

    def __init__(self,
                 config: MLoRAConfig,
                 trainable_params: Dict[str, List[torch.Tensor]]):
        lora_configs = config.lora_configs_
        tainer_config = config.trainer_config_
        for lora_config in lora_configs:
            context = TrainerContext(
                lora_config,
                tainer_config,
                trainable_params[lora_config.adapter_name_])
            self.trainer_context_[context.adapter_name_] = context

    def calc_loss(self, train_data: MultiLoraBatchData, model_output: torch.Tensor) -> torch.Tensor:
        labels = torch.tensor(train_data.batch_tokens_, dtype=torch.long)
        total_loss = None

        for lora_config in train_data.lora_batch_data_config_:
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_
            adapter_name = lora_config.adapter_name_
            vocab_size = model_output.shape[-1]
            loss_input = model_output[start_idx:end_idx][...,
                                                         :-1, :].contiguous().view(-1, vocab_size)
            loss_target = labels[start_idx:end_idx][...,
                                                    1:].contiguous().view(-1).to(loss_input.device)
            loss = self.trainer_context_[
                adapter_name].loss_fn_(loss_input, loss_target)
            print(f"    adpter: {adapter_name} loss: {loss}")
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        return total_loss

    def step(self, adapter_name: str):
        self.trainer_context_[adapter_name].step()

    def is_save_step(self, adapter_name: str) -> bool:
        return self.trainer_context_[adapter_name].is_save_step()

    def get_step_cnt(self, adapter_name: str) -> int:
        return self.trainer_context_[adapter_name].step_cnt_

    def get_trainer_context(self, adapter_name: str) -> int:
        return self.trainer_context_[adapter_name]


class Trainer:
    model_: LLMModel = None
    dispatcher_: Dispatcher = None
    multi_trainer_context_: MutiTrainerContext = None

    def __init__(self,
                 model: LLMModel,
                 dispatcher: Dispatcher,
                 config: MLoRAConfig) -> None:
        self.model_ = model
        self.dispatcher_ = dispatcher
        all_trainable_params = self.model_.get_train_paramas()
        self.multi_trainer_context_ = MutiTrainerContext(config, all_trainable_params)

    def train(self, save_step: int = 2000) -> None:
        for train_data in self.dispatcher_.train_data():
            output = self.model_.forward(train_data)

            total_loss = self.multi_trainer_context_.calc_loss(train_data, output)
            total_loss.backward()

            for lora_config in train_data.lora_batch_data_config_:
                adapter_name = lora_config.adapter_name_
                self.multi_trainer_context_.step(adapter_name)
                step_cnt = self.multi_trainer_context_.get_step_cnt(adapter_name)
                if self.multi_trainer_context_.is_save_step(adapter_name):
                    self.save_lora_model(adapter_name, f"{step_cnt}")

        # flush the grad
        for adapter_name in self.trainer_context_:
            self.trainer_context_[adapter_name].finish()
            self.save_lora_model(adapter_name)

    def save_lora_model(self, adapter_name: str, dir_suffix: str = ""):
        lora_output_dir = adapter_name
        if dir_suffix != "":
            lora_output_dir += os.sep + adapter_name + "_" + dir_suffix

        if not os.path.exists(lora_output_dir):
            os.makedirs(lora_output_dir)

        lora_weight_dict, _ = self.model_.get_lora_weight_dict(adapter_name)

        torch.save(lora_weight_dict, lora_output_dir +
                   os.sep + "adapter_model.bin")

        adapter_config = self.trainer_context_[adapter_name].export_config()
        with open(lora_output_dir + os.sep + "adapter_config.json", "w") as f:
            json.dump(adapter_config, f, indent=4)
