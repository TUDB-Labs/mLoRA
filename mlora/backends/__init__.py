from .common import BasicBackend
from .cuda import CUDABackend
from .mps import MPSBackend

import torch

_backend: BasicBackend = None


def _init_backend():
    global _backend
    if torch.cuda.is_available():
        _backend = CUDABackend()
    elif torch.backends.mps.is_available():
        _backend = MPSBackend()
    else:
        raise RuntimeError("No supported torch backends")


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
    "get_backend",
]
