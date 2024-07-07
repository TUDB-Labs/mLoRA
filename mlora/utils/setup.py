import logging
import random
from typing import List, Optional

import torch

import mlora.profiler


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    # set the logger
    log_handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        format="[%(asctime)s] m-LoRA: %(message)s",
        level=log_level,
        handlers=log_handlers,
        force=True,
    )


def setup_cuda_check():
    # check the enviroment
    if torch.cuda.is_available():
        logging.info("NVIDIA CUDA initialized successfully.")
        logging.info("Total %i GPU(s) detected." % torch.cuda.device_count())
    else:
        logging.error(
            "m-LoRA requires NVIDIA CUDA computing capacity. "
            "Please check your PyTorch installation."
        )
        exit(1)


def setup_trace_mode():
    mlora.profiler.setup_trace_mode()


def setup_metric_logger(path: str):
    mlora.profiler.metric_init(path)
