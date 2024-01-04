from mlora.modelargs import MultiLoraBatchData, LoraConfig, MixConfig
from mlora.dispatcher import Dispatcher
from mlora.MixLoRA import router_loss_factory
from mlora.model import LLMModel

from dataclasses import dataclass
from typing import Dict, List
import torch


@dataclass
class TrainConfig:
    adapter_name_: str = None
    batch_size_: int = None
    micro_batch_size_: int = None
    optimizer_name_: str = None
    learning_rate_: float = None
    momentum_: float = None

    accumulation_step_: int = None
    optimizer_: torch.optim.Optimizer = None
    loss_fn_: torch.nn.Module = None
    router_loss_fn_: torch.nn.Module = None

    def init(self, train_config: Dict[str, any], lora_config: LoraConfig) -> "TrainConfig":
        self.adapter_name_ = lora_config.adapter_name_
        self.batch_size_ = train_config["batch_size"]
        self.micro_batch_size_ = train_config["micro_batch_size"]
        self.optimizer_name_ = train_config["optim"]
        self.learning_rate_ = train_config["lr"]
        self.momentum_ = train_config.get("momentum", 0)
        if isinstance(lora_config, MixConfig):
            self.router_loss_fn_ = router_loss_factory(lora_config)

        return self

    def prepare(self, train_paramas: List[torch.Tensor]):
        if self.batch_size_ < self.micro_batch_size_ or self.batch_size_ % self.micro_batch_size_ != 0:
            raise ValueError(
                f"error batch_size {self.batch_size_} and micro batch size {self.micro_batch_size_}")
        self.accumulation_step_ = self.batch_size_ / self.micro_batch_size_
        if self.optimizer_name_ == "sgd":
            self.optimizer_ = torch.optim.SGD(
                train_paramas, lr=self.learning_rate_, momentum=self.momentum_)
        elif self.optimizer_name_ == "adamw":
            self.optimizer_ = torch.optim.AdamW(
                train_paramas, lr=self.learning_rate_)
        else:
            raise ValueError(f"unkown optimizer {self.optimizer_name_}")
        self.loss_fn_ = torch.nn.CrossEntropyLoss()


def train(dispatcher: Dispatcher,
          llm_model: LLMModel,
          configs: List[TrainConfig],
          device="cuda:0",
          save_dir: str = ".",
          save_step: int = 2000,
          verbose: bool = False):
    device = torch.device(device)
    output_router_logits = False
    train_paramas = llm_model.get_train_paramas()
    for adapter_name, config in configs.items():
        if config.router_loss_fn_ is not None:
            output_router_logits = True
        config.prepare(train_paramas[adapter_name])
    step_cnt = 0
    while not dispatcher.check_task_done():
        input: MultiLoraBatchData = dispatcher.get_train_data()
        for _, config in configs.items():
            config.optimizer_.zero_grad()

        step_cnt += 1

        output, router_outputs = llm_model.forward(
            input, output_router_logits=output_router_logits)

        labels = torch.tensor(input.batch_tokens_,
                              dtype=torch.long).to(device)

        total_loss = None
        for idx, lora_config in enumerate(input.lora_batch_data_config_):
            train_config = configs[lora_config.adapter_name_]
            start_idx = lora_config.batch_start_idx_
            end_idx = lora_config.batch_end_idx_
            loss_input = output[start_idx:end_idx][..., :-1,
                                                   :].contiguous().view(-1, llm_model.vocab_size_)
            loss_target = labels[start_idx:end_idx][...,
                                                    1:].contiguous().view(-1)
            loss = train_config.loss_fn_(
                loss_input, loss_target) / train_config.accumulation_step_
            if router_outputs is not None and len(router_outputs[idx]) > 0:
                router_loss = train_config.router_loss_fn_(
                    router_outputs[idx]) / train_config.accumulation_step_
                loss += router_loss
                if verbose:
                    print(f"    adapter: {lora_config.adapter_name_}" +
                          f" loss: {loss}")
                    print(f"{' '*(6 + len(lora_config.adapter_name_))}" +
                          f" router loss: {router_loss}")
            elif verbose:
                print(f"    adapter: {lora_config.adapter_name_} loss: {loss}")
            if total_loss is None:
                total_loss = loss
            else:
                total_loss += loss

        total_loss.backward()
        for _, config in configs.items():
            if step_cnt % config.accumulation_step_ == 0:
                config.optimizer_.step()

        if step_cnt % save_step == 0:
            llm_model.save_adapter_weight(save_dir, f"{step_cnt}")

    llm_model.save_adapter_weight(save_dir)
