import torch

from typing import Tuple


def pack_hook(to_offload: torch.Tensor) -> Tuple[torch.device, torch.Tensor]:
    return to_offload.device, to_offload.to("cpu")


def unpack_hook(to_offload_info: Tuple[torch.device, torch.Tensor]) -> torch.Tensor:
    device, to_offload = to_offload_info
    return to_offload.to(device)


def CheckpointOffloadFunction(run_function, *args):
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        outputs = run_function(*args)
    return outputs
