import contextlib
import logging

import torch

from .common import BasicBackend


class CPUBackend(BasicBackend):
    def __init__(self) -> None:
        super().__init__()

    def name(self) -> str:
        return "CPU"

    def device_name(self) -> str:
        return "cpu"

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return False

    def empty_cache(self):
        pass

    def allow_tf32(self, mode: bool):
        assert not mode, "Enabling tf32 for CPU."

    def set_rng_state(self, device: int, state: torch.Tensor):
        assert device == 0
        torch.set_rng_state(state)

    def get_rng_state(self, device: int):
        assert device == 0
        return torch.get_rng_state()

    @contextlib.contextmanager
    def fork_rng(self, rng_devices: list):
        # TODO: change to official implementation
        assert len(rng_devices) == 0
        cpu_rng_state = torch.get_rng_state()
        try:
            yield
        finally:
            torch.set_rng_state(cpu_rng_state)

    def check_available(self):
        logging.info(f"{self.name()} initialized successfully.")
        return True
