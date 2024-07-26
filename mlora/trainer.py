import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from transformers import get_scheduler

from .backends import no_cache
from .common import LLMModelOutput, Prompt
from .dispatcher import Dispatcher, DispatcherConfig, TrainTask
from .evaluator import EvaluateConfig, evaluate
from .model import LLMModel
from .prompter import Prompter
from .tasks import BasicTask, CasualTask, MultiTask, task_dict
from .tokenizer import Tokenizer


@dataclass
class TrainConfig(DispatcherConfig):
    adapter_name: str = None
    num_epochs: int = None
    batch_size: int = None
    micro_batch_size: int = None
    # optimizer settings
    # optimizer types: adamw, sgd
    optimizer_type: str = "adamw"
    learning_rate: float = None
    momentum: float = 0.0
    weight_decay: float = 0.01
    optimizer_: torch.optim.Optimizer = None
    # loraplus learning rate ratio lr_B / lr_A
    loraplus_lr_ratio: float = 1.0
    # scheduler types
    #   constant, linear, cosine, cosine_with_restarts, polynomial
    #   constant_with_warmup, inverse_sqrt, reduce_lr_on_plateau
    scheduler_type: str = "constant"
    warmup_ratio: Union[int, float] = 0
    lr_scheduler_: torch.optim.lr_scheduler.LRScheduler = None
    accumulation_step_: int = None
    training_steps_: int = 0
    task_name: str = "casual"
    task_: BasicTask = None
    group_by_length: bool = False
    # task settings
    data_path: str = None
    prompt_template: str = None
    # on-the-fly evaluation settings
    evaluate_steps: int = None
    evaluate_configs_: List[EvaluateConfig] = None

    @staticmethod
    def from_config(config: Dict[str, any]):
        batch_size = config["batch_size"]
        evaluate_steps = config.get("evaluate_steps", None)
        return TrainConfig(
            adapter_name=config["name"],
            task_name=config.get("task_name", "casual"),
            num_epochs=config["num_epochs"],
            batch_size=batch_size,
            micro_batch_size=config.get("micro_batch_size", batch_size),
            optimizer_type=config.get("optim", "adamw"),
            learning_rate=config["lr"],
            loraplus_lr_ratio=config.get("loraplus_lr_ratio", 1.0),
            momentum=config.get("momentum", 0),
            weight_decay=config.get("weight_decay", 0.01),
            scheduler_type=config.get("scheduler_type", "constant"),
            warmup_ratio=config.get("warmup_ratio", 0),
            group_by_length=config.get("group_by_length", False),
            data_path=config.get("data", None),
            prompt_template=config.get("prompt", None),
            evaluate_steps=config.get("evaluate_steps", None),
            evaluate_configs_=(
                EvaluateConfig.from_config(config) if evaluate_steps else None
            ),
        )

    def _dataload_fn(self, tokenizer: Tokenizer, **tokenizer_kwargs):
        prompter = None
        data = self.task_.loading_data(True, self.data_path)
        for idx, data_point in enumerate(data):
            if isinstance(data_point.inputs, Prompt):
                if prompter is None:
                    prompter = Prompter(self.prompt_template)
                data_point.inputs = prompter.generate_prompt(
                    instruction=data_point.inputs.instruction,
                    input=data_point.inputs.input,
                    label=data_point.inputs.label,
                )

            data_point.tokens = tokenizer.encode(data_point.inputs, **tokenizer_kwargs)
            if idx % 10000 == 0:
                logging.info(f"Encode text data: {idx}/{len(data)}")

        return data

    def dispatcher_context(self) -> Dict[str, any]:
        return {
            "adapter_name": self.adapter_name,
            "dataload_function": self._dataload_fn,
            "total_epoch_num": self.num_epochs,
            "max_train_batch_size": self.batch_size,
            "max_train_micro_batch_size": self.micro_batch_size,
            "group_by_length": self.group_by_length,
        }

    def _optimizer_grouped_parameters(self, train_paramas: Dict[str, torch.Tensor]):
        assert self.loraplus_lr_ratio >= 1.0
        if self.loraplus_lr_ratio == 1.0:
            return [
                {
                    "params": list(
                        params
                        for params in train_paramas.values()
                        if params.requires_grad
                    ),
                    "lr": self.learning_rate,
                }
            ]
        logging.info(f"Initializing {self.adapter_name} for LoRA+")
        param_groupA = []
        param_groupB = []
        for name, param in train_paramas.items():
            if not param.requires_grad:
                continue
            if "lora_B" in name or param.ndim == 1:
                param_groupB.append(param)
            else:
                param_groupA.append(param)

        return [
            {
                "params": param_groupA,
                "lr": self.learning_rate,
            },
            {
                "params": param_groupB,
                "lr": self.learning_rate * self.loraplus_lr_ratio,
            },
        ]

    def prepare(self, train_params: Dict[str, torch.Tensor]):
        # preparing for training task
        if self.task_name == "casual":
            self.task_ = CasualTask()
        elif ";" in self.task_name:
            self.task_ = MultiTask(self.task_name)
        else:
            self.task_ = task_dict[self.task_name]

        # preparing batch size and gradient accumulation
        if (
            self.batch_size < self.micro_batch_size
            or self.batch_size % self.micro_batch_size != 0
        ):
            raise ValueError(
                f"error batch_size {self.batch_size} and micro batch size {self.micro_batch_size}"
            )
        self.accumulation_step_ = self.batch_size / self.micro_batch_size
        self.training_steps_ = 0
        # preparing optimizer
        paramas_count = sum(t.numel() for t in train_params.values() if t.requires_grad)
        logging.info(f"{self.adapter_name} total trainable params: {paramas_count}")
        grouped_parameters = self._optimizer_grouped_parameters(train_params)
        if self.optimizer_type == "sgd":
            self.optimizer_ = torch.optim.SGD(
                grouped_parameters,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_type == "adamw":
            self.optimizer_ = torch.optim.AdamW(
                grouped_parameters, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"unkown optimizer {self.optimizer_type}")

    def prepare_lr_scheduler(self, len_dataset):
        if self.lr_scheduler_ is None:
            total_steps = (
                (len_dataset // self.batch_size) * self.num_epochs
                if len_dataset % self.batch_size == 0
                else (len_dataset // self.batch_size + 1) * self.num_epochs
            )
            self.lr_scheduler_ = get_scheduler(
                self.scheduler_type,
                self.optimizer_,
                self.warmup_ratio * total_steps,
                total_steps,
            )

    def step(self):
        self.training_steps_ += 1
        if self.training_steps_ % self.accumulation_step_ == 0:
            self.optimizer_.step()
            self.lr_scheduler_.step()
            self.optimizer_.zero_grad()

    def finish(self):
        self.optimizer_.step()
        self.optimizer_.zero_grad()


def save_adapter_weight(model: LLMModel, config: TrainConfig, path: str, dir_suffix=""):
    lora_output_dir = path + os.sep + config.adapter_name
    if dir_suffix != "":
        lora_output_dir += os.sep + config.adapter_name + "_" + dir_suffix

    if not os.path.exists(lora_output_dir):
        os.makedirs(lora_output_dir)

    lora_weight_dict = model.get_adapter_weight_dict(config.adapter_name)
    lora_config_dict = model.adapter_configs_[config.adapter_name].export()
    lora_config_dict["base_model_name_or_path"] = model.name_or_path_
    lora_config_dict["task_type"] = config.task_.peft_task_type

    torch.save(lora_weight_dict, lora_output_dir + os.sep + "adapter_model.bin")

    with open(lora_output_dir + os.sep + "adapter_config.json", "w") as f:
        json.dump(lora_config_dict, f, indent=4)


def _compute_loss(config_dict: Dict[str, TrainConfig], outputs: List[LLMModelOutput]):
    total_loss = None
    for output in outputs:
        adapter_name = output.adapter_name
        loss = output.loss / config_dict[adapter_name].accumulation_step_
        logging.info(f"    adapter: {adapter_name} loss: {loss}")
        if output.aux_loss:
            aux_loss = output.aux_loss / config_dict[adapter_name].accumulation_step_
            logging.info(f"    adapter: {adapter_name}  aux: {aux_loss}")
            loss += aux_loss
        if total_loss is None:
            total_loss = loss
        else:
            total_loss += loss

    return total_loss


def _perform_evaluate(
    train_configs: Dict[str, TrainConfig],
    evaluate_configs: List[EvaluateConfig],
    **kwargs,
):
    if len(evaluate_configs) > 0:
        with no_cache():
            results = evaluate(configs=evaluate_configs, **kwargs)
    else:
        results = []

    for dic in results:
        adapter_name = dic["adapter_name"]
        config = train_configs[adapter_name]
        dic["training_steps"] = config.training_steps_

    return results


def train(
    model: LLMModel,
    tokenizer: Tokenizer,
    configs: List[TrainConfig],
    max_concurrent_jobs: int = None,
    strategy: str = "optim",
    cutoff_len: Optional[int] = None,
    save_step: Optional[int] = None,
    save_dir: Optional[str] = None,
) -> None:
    if cutoff_len is None:
        cutoff_len = model.config_.max_seq_len_

    if model.config_.attn_implementation_ != "eager":
        logging.warn(
            "It is strongly recommended to train models with the `eager` attention implementation "
            f"instead of `{model.config_.attn_implementation_}`."
        )

    dispatcher = Dispatcher(
        tokenizer, configs, max_concurrent_jobs, strategy, cutoff_len
    )

    config_dict: Dict[str, TrainConfig] = {}
    for config in configs:
        config_dict[config.adapter_name] = config
        config.prepare(model.get_adapter_weight_dict(config.adapter_name))

    def task_in_callback(task: TrainTask):
        adapter_name = task.adapter_name_
        logging.info(f"Loading training task {adapter_name}")
        config = config_dict[adapter_name]
        config.prepare_lr_scheduler(len(task.train_token_data_))

    dispatcher.train_task_in_event_.register(task_in_callback)

    evaluate_results = []

    while not dispatcher.check_task_done():
        input_args = dispatcher.get_train_data()

        outputs = model.forward(input_args)

        total_loss = _compute_loss(config_dict, outputs)

        total_loss.backward()

        evaluate_configs = []

        for output in outputs:
            config = config_dict[output.adapter_name]
            config.step()

            if save_step is not None and config.training_steps_ % save_step == 0:
                save_adapter_weight(
                    model, config, save_dir, f"{config.training_steps_}"
                )

            if (
                config.evaluate_steps is not None
                and config.training_steps_ % config.evaluate_steps == 0
            ):
                evaluate_configs.extend(config.evaluate_configs_)

        evaluate_results.extend(
            _perform_evaluate(
                train_configs=config_dict,
                evaluate_configs=evaluate_configs,
                model=model,
                tokenizer=tokenizer,
                max_concurrent_jobs=max_concurrent_jobs,
                max_seq_len=cutoff_len,
            )
        )

    evaluate_configs = []

    for config in configs:
        config.finish()
        if save_dir:
            save_adapter_weight(model, config, save_dir)
        if config.evaluate_steps is not None:
            evaluate_configs.extend(config.evaluate_configs_)

    evaluate_results.extend(
        _perform_evaluate(
            train_configs=config_dict,
            evaluate_configs=evaluate_configs,
            model=model,
            tokenizer=tokenizer,
            max_concurrent_jobs=max_concurrent_jobs,
            max_seq_len=cutoff_len,
        )
    )

    if len(evaluate_results) > 0:
        if save_dir is not None:
            save_file = f"{save_dir}{os.sep}mlora_train_{int(time.time())}.json"
            with open(save_file, "w") as f:
                json.dump(evaluate_results, f, indent=4)
            logging.info(f"saving evaluation result to {save_file}")
        else:
            print(json.dumps(evaluate_results, indent=4))
