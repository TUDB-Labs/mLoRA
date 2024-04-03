import logging
import random
import torch


def check_available():
    if torch.cuda.is_available():
        logging.info('NVIDIA CUDA initialized successfully.')
        logging.info('Total %i GPU(s) detected.' % torch.cuda.device_count())
        return True
    elif torch.backends.mps.is_available():
        logging.info('Apple MPS initialized successfully.')
        return True
    else:
        logging.error(
            'm-LoRA requires CUDA or MPS computing capacity. Please check your PyTorch installation.')
        return False


def default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        raise RuntimeError("No supported torch backends")


def manual_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    else:
        raise RuntimeError("No supported torch backends")


def is_bf16_supported():
    if torch.cuda.is_available():
        return torch.cuda.is_bf16_supported()
    else:
        return False


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    else:
        raise RuntimeError("No supported torch backends")


def use_deterministic_algorithms(mode: bool):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = not mode
        torch.backends.cudnn.deterministic = mode
    elif torch.backends.mps.is_available():
        torch.use_deterministic_algorithms(mode)
    else:
        raise RuntimeError("No supported torch backends")


def cuda_allow_tf32(mode: bool):
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = mode
        torch.backends.cuda.matmul.allow_tf32 = mode
    elif mode:
        logging.warn("Enabling tf32 for non-NVIDIA devices.")


def set_rng_state(device, state):
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            return torch.cuda.set_rng_state(state)
    elif torch.backends.mps.is_available():
        return torch.mps.set_rng_state(state)
    else:
        raise RuntimeError("No supported torch backends")


def get_rng_state(device):
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            return torch.cuda.get_rng_state()
    elif torch.backends.mps.is_available():
        return torch.mps.get_rng_state()
    else:
        raise RuntimeError("No supported torch backends")


def is_initialized():
    if torch.cuda.is_available():
        return torch.cuda.is_initialized()
    elif torch.backends.mps.is_available():
        return not torch.mps._is_in_bad_fork
    else:
        raise RuntimeError("No supported torch backends")
