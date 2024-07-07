import os

import torch

from .common import BasicBackend
from .cpu import CPUBackend
from .cuda import CUDABackend
from .mps import MPSBackend

_backend: BasicBackend = None


backend_dict = {
    "CUDA": CUDABackend,
    "MPS": MPSBackend,
    "CPU": CPUBackend,
}


def _init_backend():
    global _backend
    env = os.getenv("MLORA_BACKEND_TYPE")
    if env is not None:
        env = env.upper()
        if env not in backend_dict:
            raise ValueError(f"Assigning unknown backend type {env}")
        _backend = backend_dict[env]()
    elif torch.cuda.is_available():
        _backend = CUDABackend()
    elif torch.backends.mps.is_available():
        _backend = MPSBackend()
    else:
        _backend = CPUBackend()


def get_backend() -> BasicBackend:
    if _backend is None:
        _init_backend()

    return _backend


__all__ = [
    "_backend",
    "_init_backend",
    "BasicBackend",
    "CUDABackend",
    "MPSBackend",
    "CPUBackend",
    "get_backend",
]
