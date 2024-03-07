import datasets
import logging
import random
import mlora
import torch
import json
import math
import fire
import csv

from typing import List

choices_map = ["A", "B", "C", "D"]


def format_subject(subject):
    lst = subject.split("_")
    sjt = ""
    for entry in lst:
        sjt += " " + entry
    return sjt


def format_prompt(data_point, with_answer=True):
    question = data_point["question"].strip()
    choices = "".join(
        [f"{key}. {choice}\n" for key, choice in zip(
            choices_map, data_point["choices"])]
    )
    prompt = f"{question}\n{choices}Answer:"
    if with_answer:
        prompt += " {}\n\n".format(choices_map[data_point["answer"]])
    return prompt


def prepare_data(tokenizer: mlora.Tokenizer,
                 subject: str,
                 dev_data: datasets.Dataset,
                 test_data: datasets.Dataset,
                 k_shots=5,
                 max_seq_len=2048,
                 batch_padding=True):

    sequence_lengths = []
    batch_tokens = []
    batch_labels = []
    atten_masks = []

    max_tokens_len = 0
    tokens = None
    for test_data_point in test_data:
        test_prompt = format_prompt(test_data_point, False)
        dev_prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
            format_subject(subject)
        )
        k = k_shots
        for dev_data_point in dev_data:
            k -= 1
            prompt = format_prompt(dev_data_point)
            input_ids = tokenizer.encode(
                dev_prompt + prompt + test_prompt, True, False)
            if len(input_ids) <= max_seq_len:
                tokens = input_ids
                dev_prompt += prompt
            else:
                k = 0

            if k <= 0:
                break

        max_tokens_len = max(len(tokens), max_tokens_len)
        batch_tokens.append(tokens)
        batch_labels.append(test_data_point["answer"])

    if batch_padding:
        logging.info(f"Max tokens: {max_tokens_len}/{max_seq_len}")
        if max_tokens_len < max_seq_len:
            max_seq_len = math.ceil(max_tokens_len / 8) * 8
        logging.info(f"Max sequence length: {max_seq_len}")

    for tokens in batch_tokens:
        if batch_padding:
            sequence_lengths.append(len(tokens) - 1)
            while len(tokens) < max_seq_len:
                tokens.append(tokenizer.pad_id_)
        else:
            sequence_lengths.append(-1)
        atten_masks.append(tokenizer.attention_mask(tokens))

    return sequence_lengths, batch_tokens, atten_masks, batch_labels


@torch.inference_mode()
def evaluate(subject: str,
             tokenizer: mlora.Tokenizer,
             model: mlora.LlamaModel,
             adapter_names: List[str],
             batch_size: int = 2,
             max_seq_len: int = 2048):
    # prepare data

    mmlu = datasets.load_dataset("cais/mmlu", subject)

    sequence_lengths, batch_tokens, atten_masks, batch_labels = prepare_data(
        tokenizer, subject, mmlu["dev"], mmlu["test"], 5, max_seq_len, batch_size > 1)

    # load adapters

    results = {}

    for name in adapter_names:
        results[name] = []

    # prepare for evaluate
    sequence_lengths = torch.tensor(
        sequence_lengths, dtype=torch.long, device=model.device_)

    label_indices = [0] * len(choices_map)
    for idx, text in enumerate(choices_map):
        ids = tokenizer.encode(text, False, False)
        label_indices[idx] = ids[-1]
    label_indices = torch.tensor(
        label_indices, dtype=torch.long, device=model.device_)

    start_pos = 0
    while start_pos < len(batch_tokens):
        end_pos = min(len(batch_tokens), start_pos + batch_size)
        logging.info(f"evaluation step: {start_pos}/{len(batch_tokens)}")
        bsz = end_pos - start_pos
        batch_data_config = []
        batch_start_idx = 0
        for name in adapter_names:
            batch_data_config.append(mlora.LoraBatchDataConfig(
                adapter_name_=name,
                batch_start_idx_=batch_start_idx,
                batch_end_idx_=batch_start_idx + bsz,
            ))
            batch_start_idx += bsz

        input_args = mlora.MultiLoraBatchData(
            lora_batch_data_config_=batch_data_config,
            batch_tokens_=batch_tokens[start_pos:end_pos] * len(adapter_names),
            attention_masks_=atten_masks[start_pos:end_pos] * len(adapter_names),
            gradient_checkpoint_=False,
            inference_seq_pos_=-1 if batch_size > 1 else 0,
        )

        outputs = model.forward(input_args)

        labels = torch.tensor(
            batch_labels[start_pos:end_pos], dtype=torch.long, device=model.device_)

        for output in outputs:
            logits = output.logits
            logits = logits[torch.arange(
                bsz, device=logits.device), sequence_lengths[start_pos:end_pos]]
            logits = logits[:, label_indices]
            logits = logits.softmax(-1).argmax(-1)
            result = (logits == labels).int().tolist()
            results[output.adapter_name].extend(result)

        for name, result in results.items():
            acc = sum(result) / len(result)
            logging.info(f"    {name} accuracy: {acc}")

        start_pos = end_pos

    return results


mmlu_subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}


mmlu_categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}


model_dtypes = {
    "4bit": {"bits": 4, "load_dtype": torch.float32},
    "8bit": {"bits": 8, "load_dtype": torch.float32},
    "16bit": {"load_dtype": torch.bfloat16},
}


def do_evaluate(model_name: str,
                model_dtype: str,
                adapter_names: List[str],
                batch_size: int = 2,
                device: str = "cuda:0",
                output: str = "mmlu_scores.csv"):
    tokenizer = mlora.Tokenizer(model_name)
    model = mlora.LlamaModel.from_pretrained(
        model_name, device=device, **model_dtypes[model_dtype])
    for name in adapter_names:
        logging.info(f"Loading adapter {name}")
        model.load_adapter_weight(name)

    csv_data = [["mmlu_categories", "mmlu_subcategories",
                 "adapter_name", "acc_score"]]
    for subject, subcategory in mmlu_subcategories.items():
        logging.info(f"Performing MMLU/{subject} Benchmark")
        results = evaluate(subject, tokenizer, model,
                           adapter_names, batch_size, model.max_seq_len_)
        category = None
        for category_name, subcategory_names in mmlu_categories.items():
            if subcategory[-1] in subcategory_names:
                category = category_name
        for name, result in results.items():
            acc = sum(result) / len(result)
            csv_data.append([category, subject, name, acc])
        with open(output, "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(csv_data)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def main(config: str):
    setup_seed(66)
    log_handlers = [logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] m-LoRA: %(message)s',
                        level=logging.INFO,
                        handlers=log_handlers,
                        force=True)
    with open(config, 'r', encoding='utf8') as fp:
        mmlu_config = json.load(fp)
    do_evaluate(**mmlu_config)


if __name__ == "__main__":
    fire.Fire(main)
