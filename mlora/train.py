from mlora.modelargs import MultiLoraBatchData, LoraConfig, MixConfig
from mlora.dispatcher import Dispatcher
from mlora.mix_lora import router_loss_factory
from mlora.tasks import train_task_factory
from mlora.model import LLMModel

from typing import Dict, List
import logging
import torch
import json
import os


class TrainConfig:
    def __init__(self, model: LLMModel,
                 train_config: Dict[str, any],
                 lora_config: LoraConfig,
                 lora_weight: Dict[str, torch.Tensor]):
        self.adapter_name_ = lora_config.adapter_name_
        self.batch_size_ = train_config["batch_size"]
        self.micro_batch_size_ = train_config["micro_batch_size"]
        self.optimizer_name_ = train_config["optim"]
        self.learning_rate_ = train_config["lr"]
        self.momentum_ = train_config.get("momentum", 0)
        self.router_loss_fn_ = router_loss_factory(
            lora_config) if isinstance(lora_config, MixConfig) else None
        self.accumulation_step_: int = None
        self.optimizer_: torch.optim.Optimizer = None
        self.task = train_task_factory(model, train_config, lora_weight)
        train_config["dataloader"] = self.task.dataload_function
        if "task_type" in train_config and "data" not in train_config:
            train_config["data"] = train_config["task_type"]

    def prepare(self, train_paramas: List[torch.Tensor]):
        if self.batch_size_ < self.micro_batch_size_ or self.batch_size_ % self.micro_batch_size_ != 0:
            raise ValueError(
                f"error batch_size {self.batch_size_} and micro batch size {self.micro_batch_size_}")
        self.accumulation_step_ = self.batch_size_ / self.micro_batch_size_
        train_paramas.extend(self.task.state_dict().values())
        paramas_count = sum(t.numel()
                            for t in train_paramas if t.requires_grad)
        logging.info(
            f"{self.adapter_name_} total trainable params: {paramas_count}")
        if self.optimizer_name_ == "sgd":
            self.optimizer_ = torch.optim.SGD(
                train_paramas, lr=self.learning_rate_, momentum=self.momentum_)
        elif self.optimizer_name_ == "adamw":
            self.optimizer_ = torch.optim.AdamW(
                train_paramas, lr=self.learning_rate_)
        else:
            raise ValueError(f"unkown optimizer {self.optimizer_name_}")


def save_adapter_weight(model: LLMModel, config: TrainConfig, path: str, dir_suffix=""):
    lora_output_dir = path + os.sep + config.adapter_name_
    if dir_suffix != "":
        lora_output_dir += os.sep + \
            config.adapter_name_ + "_" + dir_suffix

    if not os.path.exists(lora_output_dir):
        os.makedirs(lora_output_dir)

    lora_weight_dict = model.get_lora_weight_dict(config.adapter_name_)
    lora_config_dict = model.adapter_configs_[config.adapter_name_].export()

    extra_paramas = config.task.state_dict()
    if len(extra_paramas) > 0:
        lora_weight_dict.update(extra_paramas)

    torch.save(lora_weight_dict, lora_output_dir +
               os.sep + "adapter_model.bin")

    with open(lora_output_dir + os.sep + "adapter_config.json", "w") as f:
        json.dump(lora_config_dict, f, indent=4)


def train(dispatcher: Dispatcher,
          model: LLMModel,
          configs: List[TrainConfig],
          save_dir: str = ".",
          save_step: int = 2000) -> None:
    config_dict = {}
    for config in configs:
        config_dict[config.adapter_name_] = config

    output_router_logits = False
    train_paramas = model.get_train_paramas()
    for config in configs:
        if config.router_loss_fn_ is not None:
            output_router_logits = True
        config.prepare(train_paramas[config.adapter_name_])

    step_cnt = 0
    while not dispatcher.check_task_done():
        input: MultiLoraBatchData = dispatcher.get_train_data()
        input.batch_tokens_ = torch.tensor(
            input.batch_tokens_, dtype=torch.long, device=model.device_)
        input.output_router_logits_ = output_router_logits

        for config in configs:
            config.optimizer_.zero_grad()

        step_cnt += 1

        output, router_outputs = model.forward(input)

        total_loss = None
        for idx, lora_config in enumerate(input.lora_batch_data_config_):
            train_config = config_dict[lora_config.adapter_name_]
            train_task = train_config.task
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_
            logits = train_task.forward(output[start_idx:end_idx])
            loss = train_task.loss(input.batch_tokens_[start_idx:end_idx],
                                   logits, input.batch_labels_[start_idx:end_idx])
            loss = loss / train_config.accumulation_step_
            if router_outputs is not None and len(router_outputs[idx]) > 0:
                router_loss = train_config.router_loss_fn_(router_outputs[idx])
                router_loss = router_loss / train_config.accumulation_step_
                loss += router_loss
                logging.info(f"    adapter: {lora_config.adapter_name_}" +
                             f" loss: {loss}")
                logging.info(f"{' '*(6 + len(lora_config.adapter_name_))}" +
                             f" router loss: {router_loss}")
            else:
                logging.info(
                    f"    adapter: {lora_config.adapter_name_} loss: {loss}")
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        total_loss.backward()
        for config in configs:
            if step_cnt % config.accumulation_step_ == 0:
                config.optimizer_.step()
            if step_cnt % save_step == 0:
                save_adapter_weight(model, config, save_dir, f"{step_cnt}")

    for config in configs:
        save_adapter_weight(model, config, save_dir)
