import os
from typing import Optional

import torch

from .common import BasicBackend
from .cpu import CPUBackend
from .cuda import CUDABackend
from .mps import MPSBackend

_backend: Optional[BasicBackend] = None


backend_dict = {
    "CUDA": CUDABackend,
    "MPS": MPSBackend,
    "CPU": CPUBackend,
}


def _init_backend() -> BasicBackend:
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


def get_backend() -> BasicBackend:
    global _backend
    if _backend is None:
        _backend = _init_backend()

    return _backend


__all__ = [
    "BasicBackend",
    "CUDABackend",
    "MPSBackend",
    "CPUBackend",
    "get_backend",
]
