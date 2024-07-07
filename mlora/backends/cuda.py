import torch

from .common import BasicBackend


class CUDABackend(BasicBackend):
    def __init__(self) -> None:
        super().__init__()
        torch.cuda.init()

    def name(self) -> str:
        return "NVIDIA CUDA"

    def device_name(self) -> str:
        return "cuda"

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
