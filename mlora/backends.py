from mlora.utils import NoneContexts
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

    def autocast(self, **kwargs):
        return NoneContexts()


class CUDABackend(BasicBackend):
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


class MPSBackend(BasicBackend):
    def name(self) -> str:
        return "APPLE MPS"

    def device_name(self) -> str:
        return 'mps'

    def is_available(self) -> bool:
        return torch.backends.mps.is_available()

    def is_initialized(self) -> bool:
        # TODO: change to official implementation
        return not torch.mps._is_in_bad_fork

    def is_bf16_supported(self) -> bool:
        # TODO: change to official implementation
        return False

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
