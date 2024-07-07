import contextlib

import torch

from .common import BasicBackend

_mps_bf16_supported = None


class MPSBackend(BasicBackend):
    def __init__(self) -> None:
        super().__init__()

    def name(self) -> str:
        return "APPLE MPS"

    def device_name(self) -> str:
        return "mps"

    def is_available(self) -> bool:
        return torch.backends.mps.is_available()

    def is_initialized(self) -> bool:
        # TODO: change to official implementation
        return not torch.mps._is_in_bad_fork()

    def is_bf16_supported(self) -> bool:
        # TODO: change to official implementation
        global _mps_bf16_supported
        if _mps_bf16_supported is None:
            try:
                torch.ones(5, dtype=torch.bfloat16, device="mps")
                _mps_bf16_supported = True
            except TypeError:
                _mps_bf16_supported = False

        return _mps_bf16_supported

    def manual_seed(self, seed: int):
        super().manual_seed(seed)
        torch.mps.manual_seed(seed)

    def empty_cache(self):
        torch.mps.empty_cache()

    def allow_tf32(self, mode: bool):
        assert not mode, "Enabling tf32 for MPS devices."

    def set_rng_state(self, device: int, state: torch.Tensor):
        assert device == 0
        return torch.mps.set_rng_state(state)

    def get_rng_state(self, device: int):
        assert device == 0
        return torch.mps.get_rng_state()

    @contextlib.contextmanager
    def fork_rng(self, rng_devices: list):
        # TODO: change to official implementation
        assert len(rng_devices) == 1 and rng_devices[0] == 0
        cpu_rng_state = torch.get_rng_state()
        device_rng_states = torch.mps.get_rng_state()
        try:
            yield
        finally:
            torch.set_rng_state(cpu_rng_state)
            torch.mps.set_rng_state(device_rng_states)

    def autocast(self, **kwargs):
        # TODO: change to official implementation
        # running with compatible mode
        return torch.cuda.amp.autocast(**kwargs)
