import gc
import os

import torch

from .common import BasicBackend
from .cpu import CPUBackend
from .cuda import CUDABackend
from .mps import MPSBackend

backend_dict = {
    "CUDA": CUDABackend,
    "MPS": MPSBackend,
    "CPU": CPUBackend,
}


def _init_backend():
    env = os.getenv("MLORA_BACKEND_TYPE")
    if env is not None:
        env = env.upper()
        if env not in backend_dict:
            raise ValueError(f"Assigning unknown backend type {env}")
        return backend_dict[env]()
    elif torch.cuda.is_available():
        return CUDABackend()
    elif torch.backends.mps.is_available():
        return MPSBackend()
    else:
        return CPUBackend()


backend: BasicBackend = _init_backend()


class no_cache(object):
    def __enter__(self):
        backend.empty_cache()
        gc.collect()
        return self

    def __exit__(self, type, value, traceback):
        backend.empty_cache()
        gc.collect()


__all__ = [
    "BasicBackend",
    "CUDABackend",
    "MPSBackend",
    "CPUBackend",
    "backend",
    "no_cache",
]
