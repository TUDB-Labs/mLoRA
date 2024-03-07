from evaluators.arc.evaluate import arc_easy_evaluate, arc_challenge_evaluate
from evaluators.boolq.evaluate import boolq_evaluate
from evaluators.obqa.evaluate import obqa_evaluate
from evaluators.piqa.evaluate import piqa_evaluate
from typing import List

import logging
import random
import torch
import mlora
import json
import fire
import os


evaluators = {
    "ARC-E": arc_easy_evaluate,
    "ARC-C": arc_challenge_evaluate,
    "BOOLQ": boolq_evaluate,
    "OBQA": obqa_evaluate,
    "PIQA": piqa_evaluate,
}


model_dtypes = {
    "4bit": {"bits": 4, "load_dtype": torch.float32},
    "8bit": {"bits": 8, "load_dtype": torch.float32},
    "16bit": {"load_dtype": torch.bfloat16},
}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def evaluate_bootstrap(dataset: str,
                       model_name: str,
                       model_dtype: str,
                       adapter_names: List[str],
                       batch_size: int = 2,
                       device: str = "cuda:0",
                       seed: int = 66):
    log_handlers = [logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] m-LoRA: %(message)s',
                        level=logging.INFO,
                        handlers=log_handlers,
                        force=True)

    if torch.cuda.is_available():
        logging.info('NVIDIA CUDA initialized successfully.')
        logging.info('Total %i GPU(s) detected.' % torch.cuda.device_count())
    else:
        logging.error(
            'm-LoRA requires NVIDIA CUDA computing capacity. Please check your PyTorch installation.')
        exit(-1)

    setup_seed(seed)

    tokenizer = mlora.Tokenizer(model_name)
    model = mlora.LlamaModel.from_pretrained(
        model_name, device=device, **model_dtypes[model_dtype])
    for name in adapter_names:
        logging.info(f"Loading adapter {name}")
        model.load_adapter_weight(name)

    torch.cuda.empty_cache()

    return evaluators[dataset](tokenizer=tokenizer,
                               model=model,
                               adapter_names=adapter_names,
                               batch_size=batch_size,
                               max_seq_len=model.max_seq_len_)


def main(config_file: str):
    with open(config_file, 'r', encoding='utf8') as fp:
        config_obj = json.load(fp)

    if not isinstance(config_obj, list):
        config_obj = [config_obj]

    results = []
    file_path = os.path.dirname(os.path.abspath(__file__))
    for config in config_obj:
        result = evaluate_bootstrap(**config)
        results.extend(result)
        with open(f"{file_path}{os.sep}evaluate_result.json", "w") as f:
            json.dump(results, f, indent=4)

    for result in results:
        for key, value in result.items():
            logging.info(f"{key}: {value}")


if __name__ == "__main__":
    fire.Fire(main)
