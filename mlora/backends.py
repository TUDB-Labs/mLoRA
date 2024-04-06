from mlora.utils import NoneContexts
import contextlib
import logging
import random
import torch


class BasicBackend:
    def name(self) -> str:
        pass

    def device_name(self) -> str:
        pass

    def is_available(self) -> bool:
        pass

    def is_initialized(self) -> bool:
        pass

    def is_bf16_supported(self) -> bool:
        pass

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
            devices=rng_devices, device_type=self.device_name())

    def autocast(self, **kwargs):
        return NoneContexts()

    def check_available(self):
        if not self.is_available():
            logging.error(f"{self.name()} not available.")
            return False
        if not self.is_initialized():
            logging.error(f"{self.name()} not initialized.")
            return False
        logging.info(f'{self.name()} initialized successfully.')
        return True


class CUDABackend(BasicBackend):
    def __init__(self) -> None:
        super().__init__()
        torch.cuda.init()

    def name(self) -> str:
        return "NVIDIA CUDA"

    def device_name(self) -> str:
        return 'cuda'

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def is_initialized(self) -> bool:
        return torch.cuda.is_initialized()

    def is_bf16_supported(self) -> bool:
        return torch.cuda.is_bf16_supported()

    def manual_seed(self, seed: int):
        super().manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def empty_cache(self):
        torch.cuda.empty_cache()

    def use_deterministic_algorithms(self, mode: bool):
        torch.backends.cudnn.benchmark = not mode
        torch.backends.cudnn.deterministic = mode

    def allow_tf32(self, mode: bool):
        torch.backends.cudnn.allow_tf32 = mode
        torch.backends.cuda.matmul.allow_tf32 = mode

    def set_rng_state(self, device, state):
        with torch.cuda.device(device):
            return torch.cuda.set_rng_state(state)

    def get_rng_state(self, device):
        with torch.cuda.device(device):
            return torch.cuda.get_rng_state()

    def autocast(self, **kwargs):
        return torch.cuda.amp.autocast(**kwargs)


_mps_bf16_supported = None


class MPSBackend(BasicBackend):
    def __init__(self) -> None:
        super().__init__()
        torch.mps.set_per_process_memory_fraction(1.0)

    def name(self) -> str:
        return "APPLE MPS"

    def device_name(self) -> str:
        return 'mps'

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
