import logging
import random

import torch
from transformers.utils import is_torch_bf16_available_on_device

from mlora.utils import NoneContexts


class BasicBackend:
    def name(self) -> str:
        raise NotImplementedError()

    def device_name(self) -> str:
        raise NotImplementedError()

    def default_device_name(self) -> str:
        return self.device_name()

    def is_available(self) -> bool:
        raise NotImplementedError()

    def is_initialized(self) -> bool:
        raise NotImplementedError()

    def is_bf16_supported(self) -> bool:
        return is_torch_bf16_available_on_device(self.device_name())

    def manual_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)

    def empty_cache(self):
        raise NotImplementedError()

    def use_deterministic_algorithms(self, mode: bool):
        torch.use_deterministic_algorithms(mode)

    def allow_tf32(self, mode: bool):
        raise NotImplementedError()

    def set_rng_state(self, device, state):
        raise NotImplementedError()

    def get_rng_state(self, device):
        raise NotImplementedError()

    def fork_rng(self, rng_devices: list):
        return torch.random.fork_rng(
            devices=rng_devices, device_type=self.device_name()
        )

    def autocast(self, **kwargs):
        return NoneContexts()

    def init_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(tensor)

    def index_fill(
        self, input: torch.Tensor, dim: int, index: torch.Tensor, value: torch.Tensor
    ):
        input.index_fill_(dim, index, value)

    def index_copy(
        self, input: torch.Tensor, dim: int, index: torch.Tensor, source: torch.Tensor
    ):
        input.index_copy_(dim, index, source)

    def check_available(self):
        if not self.is_available():
            logging.error(f"{self.name()} not available.")
            return False
        if not self.is_initialized():
            logging.error(f"{self.name()} not initialized.")
            return False
        logging.info(f"{self.name()} initialized successfully.")
        return True
