import mlora
import torch
import evaluate
import datasets
from typing import List
from dataclasses import dataclass


def sst2_dataloader(data_point):
    return data_point["sentence"], [int(data_point["label"])]


def mrpc_dataloader(data_point):
    return data_point["sentence1"] + " </s> " + data_point["sentence2"], [int(data_point["label"])]


dataloader = {
    "glue:sst2": sst2_dataloader,
    "glue:mrpc": mrpc_dataloader
}


@dataclass
class EvaluateConfig:
    adapter_name_: str = None
    task_type_: str = None
    # Do not set these manually
    score_: torch.nn.Linear = None
    data_: datasets.Dataset = None
    metric_: datasets.Metric = None
    batch_start_idx_: int = -1
    batch_end_idx_: int = -1

    def init_task(self):
        if ':' in self.task_type_:
            result = self.task_type_.split(':')
            self.data_ = datasets.load_dataset(
                result[0], result[1])["test"]
            self.metric_ = evaluate.load(result[0], result[1])
        else:
            self.data_ = datasets.load_dataset(self.task_type_)["test"]
            self.metric_ = evaluate.load(self.task_type_)


@torch.inference_mode()
def glue_evaluate(model: mlora.LLMModel,
                  tokenizer: mlora.Tokenizer,
                  configs: List[EvaluateConfig],
                  batch_size: int = 16,
                  batch_seq_len: int = 128,
                  device: str = "cuda:0"):
    device = torch.device(device)
    max_iterations = 0
    for config in configs:
        config.init_task()
        if len(config.data_) > max_iterations:
            max_iterations = len(config.data_)
    start_idx = 0
    while start_idx < max_iterations:
        print(f"Progress: {start_idx}/{max_iterations}")
        end_idx = start_idx + batch_size
        batch_data_config = []
        current_tasks = []
        batch_tokens = []
        batch_labels = []
        tokens_len = []
        for config in configs:
            if start_idx >= len(config.data_):
                continue
            config.batch_start_idx_ = len(batch_tokens)
            for idx in range(start_idx, end_idx):
                if idx >= len(config.data_):
                    break
                text, label = dataloader[config.task_type_](config.data_[idx])
                batch_labels.append(label)
                tokens = tokenizer.encode(data=text, bos=True, eos=True)
                if len(tokens) > batch_seq_len:
                    tokens = tokens[:batch_seq_len]
                tokens_len.append(len(tokens))
                while len(tokens) < batch_seq_len:
                    tokens.append(tokenizer.pad_id_)
                batch_tokens.append(tokens)
            config.batch_end_idx_ = len(batch_tokens)
            current_tasks.append(config)
            batch_data_config.append(mlora.LoraBatchDataConfig(adapter_name_=config.adapter_name_,
                                     batch_start_idx_=config.batch_start_idx_, batch_end_idx_=config.batch_end_idx_))
        if len(current_tasks) == 0:
            break
        batch_tokens = torch.tensor(
            batch_tokens, dtype=torch.long, device=device)
        input_data = mlora.MultiLoraBatchData(
            lora_batch_data_config_=batch_data_config,
            batch_seq_len_=batch_seq_len,
            expand_side_=["right"]*batch_tokens.shape[0],
            batch_tokens_=batch_tokens,
            tokens_len_without_pad_=tokens_len,
            checkpoint_recompute_=False,
            inference_seq_pos_=-1,
            skip_lm_head_=True)
        outputs, _ = model.forward(input_data)
        for task in current_tasks:
            input_ids = batch_tokens[task.batch_start_idx_:task.batch_end_idx_]
            logits = outputs[task.batch_start_idx_:task.batch_end_idx_]
            labels = torch.tensor(
                batch_labels[task.batch_start_idx_:task.batch_end_idx_], dtype=torch.long)
            bsz = input_ids.shape[0]
            logits = task.score_(logits.to(torch.float32))
            sequence_lengths = (
                torch.eq(input_ids, tokenizer.pad_id_).int().argmax(-1) - 1).to(logits.device)
            pooled_logits = logits[torch.arange(
                bsz, device=logits.device), sequence_lengths]
            task.metric_.add_batch(
                predictions=torch.argmax(pooled_logits, dim=-1).int(), references=labels.view(-1))
        start_idx = end_idx


base_model = "/mnt/g/LLM/Models/llama-7b-hf"
device = "cuda:0"

model = mlora.LlamaModel.from_pretrained(
    base_model, device=device, load_dtype=torch.bfloat16)
tokenizer = mlora.Tokenizer(base_model)

configs = []
model.load_adapter_weight("mrpc_lora_0")
config = EvaluateConfig(adapter_name_="mrpc_lora_0", task_type_="glue:mrpc")
config.score_ = torch.nn.Linear(
    model.dim_, 2, bias=False, dtype=torch.float32, device=device)
with torch.no_grad():
    config.score_.weight.copy_(torch.load(
        "mrpc_lora_0/extra_paramas.bin", map_location=device)["classifier"])
configs.append(config)

model.load_adapter_weight("mrpc_lora_1")
config = EvaluateConfig(adapter_name_="mrpc_lora_1", task_type_="glue:mrpc")
config.score_ = torch.nn.Linear(
    model.dim_, 2, bias=False, dtype=torch.float32, device=device)
with torch.no_grad():
    config.score_.weight.copy_(torch.load(
        "mrpc_lora_1/extra_paramas.bin", map_location=device)["classifier"])
configs.append(config)

glue_evaluate(model, tokenizer, configs, 16, 512, device)
for config in configs:
    print(f"{config.adapter_name_}: {config.metric_.compute()}")
