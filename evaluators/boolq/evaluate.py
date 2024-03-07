import datasets
import logging
import mlora
import torch
import math

from typing import List

choices_map = ["true", "false"]
choices2id = {text: idx for idx, text in enumerate(choices_map)}


def prepare_data(tokenizer: mlora.Tokenizer,
                 data: datasets.Dataset,
                 max_seq_len=2048,
                 batch_padding=True):

    sequence_lengths = []
    batch_tokens = []
    batch_labels = []
    atten_masks = []

    max_tokens_len = 0
    tokens = None
    for data_point in data:
        prompt_str = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        prompt_str += "## Instruction:\nPlease answer the following question with true or false, question: " + \
            f"{data_point['question']}?\n\n" + "Answer format: true/false\n\n"
        prompt_str += "## Response:\nThe correct answer is "
        tokens = tokenizer.encode(prompt_str, bos=True, eos=False)
        max_tokens_len = max(len(tokens), max_tokens_len)
        batch_tokens.append(tokens)
        batch_labels.append(
            choices2id["true" if data_point["answer"] else "false"])

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
def boolq_evaluate(
        tokenizer: mlora.Tokenizer,
        model: mlora.LlamaModel,
        adapter_names: List[str],
        batch_size: int = 2,
        max_seq_len: int = 2048):
    # prepare data

    boolq = datasets.load_dataset("google/boolq")

    sequence_lengths, batch_tokens, atten_masks, batch_labels = prepare_data(
        tokenizer, boolq["validation"], max_seq_len, batch_size > 1)

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
        logging.info(f"BoolQ evaluation step: {start_pos}/{len(batch_tokens)}")
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
            attention_masks_=atten_masks[start_pos:end_pos] *
            len(adapter_names),
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

    final_results = []
    for name, result in results.items():
        acc = sum(result) / len(result)
        final_results.append({
            "adapter_name": name,
            "dataset": "BoolQ",
            "metric": "accuracy",
            "value": acc,
        })

    return final_results
