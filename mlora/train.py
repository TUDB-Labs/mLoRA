from mlora.modelargs import MultiLoraBatchData, LoraConfig, MixConfig
from mlora.dispatcher import Dispatcher
from mlora.mix_lora import router_loss_factory
from mlora.prompter import Prompter
from mlora.model import LLMModel

from typing import Dict, List
import logging
import torch
import os


class BasicTask():
    def __init__(self) -> None:
        self.label_dtype_: torch.dtype = torch.long
        self.loss_fn_: torch.nn.Module = None
        self.num_labels_: int = -1

    def dataload_function(self, data_point):
        return None, None

    def extra_paramas(self):
        return {}

    def compute_loss(self, input: MultiLoraBatchData, output: torch.Tensor,
                     start_idx: int, end_idx: int) -> torch.Tensor:
        pass


class SelfSupervised(BasicTask):
    def __init__(self, model: LLMModel, file_name: str) -> None:
        super().__init__()
        self.label_dtype_ = torch.long
        self.num_labels_ = model.vocab_size_
        self.prompter_ = Prompter(file_name)

    def dataload_function(self, data_point):
        return self.prompter_.generate_prompt(
            data_point["instruction"],
            data_point.get("input", None),
            data_point.get("output", None)), None, {"bos": True, "eos": False}

    def compute_loss(self, input: MultiLoraBatchData, output: torch.Tensor,
                     start_idx: int, end_idx: int) -> torch.Tensor:
        labels = torch.tensor(input.batch_labels_[
                              start_idx:end_idx], dtype=self.label_dtype_, device=output.device)
        logits = output[start_idx:end_idx]
        loss_fn = torch.nn.CrossEntropyLoss()
        return loss_fn(logits.contiguous().view(-1, self.num_labels_), labels.contiguous().view(-1))


class Classification(BasicTask):
    def __init__(self, model: LLMModel, label_dtype: torch.dtype, num_labels: int) -> None:
        super().__init__()
        self.label_dtype_ = label_dtype
        self.num_labels_ = num_labels
        self.pad_id_ = model.pad_token_id_
        self.score_ = torch.nn.Linear(
            model.dim_, self.num_labels_, bias=False, dtype=torch.float32, device=model.device_)

    def extra_paramas(self):
        return {"classifier": self.score_.weight}

    def compute_loss(self, input: MultiLoraBatchData, output: torch.Tensor,
                     start_idx: int, end_idx: int) -> torch.Tensor:
        input_ids = torch.tensor(input.batch_tokens_[
            start_idx:end_idx], dtype=self.label_dtype_, device=output.device)
        labels = torch.tensor(input.batch_labels_[
                              start_idx:end_idx], dtype=self.label_dtype_, device=output.device)
        logits = output[start_idx:end_idx]
        batch_size = input_ids.shape[0]
        logits = self.score_(logits.to(torch.float32))
        if self.pad_id_ is None:
            sequence_lengths = -1
        else:
            sequence_lengths = (
                torch.eq(input_ids, self.pad_id_).int().argmax(-1) - 1).to(logits.device)
        pooled_logits = logits[torch.arange(
            batch_size, device=logits.device), sequence_lengths]
        if self.num_labels_ == 1:
            loss_fn = torch.nn.MSELoss()
            return loss_fn(pooled_logits.squeeze(), labels.squeeze())
        elif self.num_labels_ > 1 and self.label_dtype_ == torch.long:
            loss_fn = torch.nn.CrossEntropyLoss()
            return loss_fn(pooled_logits.view(-1, self.num_labels_), labels.view(-1))
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss()
            return loss_fn(pooled_logits, labels)


class GlueSst2(Classification):
    def __init__(self, model: LLMModel) -> None:
        super().__init__(model, label_dtype=torch.long, num_labels=2)

    @staticmethod
    def dataload_function(data_point):
        return data_point["sentence"], [int(data_point["label"])], {"bos": True, "eos": False}


class GlueMrpc(Classification):
    def __init__(self, model: LLMModel) -> None:
        super().__init__(model, label_dtype=torch.long, num_labels=2)

    @staticmethod
    def dataload_function(data_point):
        return (data_point["sentence1"] + "<s>" + data_point["sentence2"],
                [int(data_point["label"])], {"bos": True, "eos": False})


classification_task_dict = {
    "glue:sst2": GlueSst2,
    "glue:mrpc": GlueMrpc,
}


class TrainConfig:
    def __init__(self, model: LLMModel, train_config: Dict[str, any], lora_config: LoraConfig):
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
        task_type = train_config.get("task_type", "self-supervised")
        if task_type == "self-supervised":
            self.task = SelfSupervised(model, train_config["prompt"])
        else:
            self.task = classification_task_dict[task_type](model)
            if "data" not in train_config:
                train_config["data"] = task_type
        train_config["dataloader"] = self.task.dataload_function

    def prepare(self, train_paramas: List[torch.Tensor]):
        if self.batch_size_ < self.micro_batch_size_ or self.batch_size_ % self.micro_batch_size_ != 0:
            raise ValueError(
                f"error batch_size {self.batch_size_} and micro batch size {self.micro_batch_size_}")
        self.accumulation_step_ = self.batch_size_ / self.micro_batch_size_
        train_paramas.extend(self.task.extra_paramas().values())
        if self.optimizer_name_ == "sgd":
            self.optimizer_ = torch.optim.SGD(
                train_paramas, lr=self.learning_rate_, momentum=self.momentum_)
        elif self.optimizer_name_ == "adamw":
            self.optimizer_ = torch.optim.AdamW(
                train_paramas, lr=self.learning_rate_)
        else:
            raise ValueError(f"unkown optimizer {self.optimizer_name_}")

    def compute_loss(self, input: MultiLoraBatchData, output: torch.Tensor,
                     start_idx: int, end_idx: int) -> torch.Tensor:
        return self.task.compute_loss(input, output, start_idx, end_idx) / self.accumulation_step_

    def router_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        return self.router_loss_fn_(router_logits) / self.accumulation_step_

    def step(self):
        return self.optimizer_.step()

    def save(self, path: str, dir_suffix=""):
        lora_output_dir = path + os.sep + self.adapter_name_
        if dir_suffix != "":
            lora_output_dir += os.sep + \
                self.adapter_name_ + "_" + dir_suffix

        if not os.path.exists(lora_output_dir):
            os.makedirs(lora_output_dir)

        extra_paramas = self.task.extra_paramas()
        if len(extra_paramas) > 0:
            torch.save(extra_paramas, lora_output_dir +
                       os.sep + "extra_paramas.bin")


def train(dispatcher: Dispatcher,
          model: LLMModel,
          configs: List[TrainConfig],
          device="cuda:0",
          save_dir: str = ".",
          save_step: int = 2000) -> None:
    device = torch.device(device)
    config_dict = {}
    for config in configs:
        config_dict[config.adapter_name_] = config

    output_router_logits = False
    train_paramas = model.get_train_paramas()
    classification_task = False
    for config in configs:
        if isinstance(config.task, SelfSupervised):
            assert not classification_task
        else:
            classification_task = True
        if config.router_loss_fn_ is not None:
            output_router_logits = True
        config.prepare(train_paramas[config.adapter_name_])

    step_cnt = 0
    while not dispatcher.check_task_done():
        input: MultiLoraBatchData = dispatcher.get_train_data()
        input.output_router_logits_ = output_router_logits
        input.skip_lm_head_ = classification_task

        for config in configs:
            config.optimizer_.zero_grad()

        step_cnt += 1

        output, router_outputs = model.forward(input)

        total_loss = None
        for idx, lora_config in enumerate(input.lora_batch_data_config_):
            train_config = config_dict[lora_config.adapter_name_]
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_
            loss = train_config.compute_loss(input, output, start_idx, end_idx)
            if router_outputs is not None and len(router_outputs[idx]) > 0:
                router_loss = train_config.router_loss(router_outputs[idx])
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
                config.step()
            if step_cnt % save_step == 0:
                config.save(save_dir, f"{step_cnt}")

        if step_cnt % save_step == 0:
            model.save_adapter_weight(save_dir, f"{step_cnt}")

    for config in configs:
        config.save(save_dir)

    model.save_adapter_weight(save_dir)
