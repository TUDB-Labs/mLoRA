import logging
import random

import torch

from mlora.utils import NoneContexts


class BasicBackend:
    def name(self) -> str:
        return "none"

    def device_name(self) -> str:
        return "null"

    def is_available(self) -> bool:
        return False

    def is_initialized(self) -> bool:
        return False

    def is_bf16_supported(self) -> bool:
        return False

    def manual_seed(self, seed: int):
        random.seed(seed)
        torch.manual_seed(seed)

    def empty_cache(self):
        pass

    def use_deterministic_algorithms(self, mode: bool):
        torch.use_deterministic_algorithms(mode)

    def allow_tf32(self, mode: bool):
        pass

    def set_rng_state(self, device, state):
        pass

    def get_rng_state(self, device):
        pass

    def fork_rng(self, rng_devices: list):
        return torch.random.fork_rng(
            devices=rng_devices, device_type=self.device_name()
        )

    def autocast(self, **kwargs):
        return NoneContexts()

    def check_available(self):
        if not self.is_available():
            logging.error(f"{self.name()} not available.")
            return False
        if not self.is_initialized():
            logging.error(f"{self.name()} not initialized.")
            return False
        logging.info(f"{self.name()} initialized successfully.")
        return True
