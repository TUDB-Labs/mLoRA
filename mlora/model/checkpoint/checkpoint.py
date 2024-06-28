from typing import Callable, Tuple

import torch


def __pack_hook(to_offload: torch.Tensor) -> Tuple[torch.device, torch.Tensor]:
    return to_offload.device, to_offload.to("cpu")


def __unpack_hook(to_offload_info: Tuple[torch.device, torch.Tensor]) -> torch.Tensor:
    device, to_offload = to_offload_info
    return to_offload.to(device)


def CheckpointOffloadFunction(run_function: Callable, *args):
    with torch.autograd.graph.saved_tensors_hooks(__pack_hook, __unpack_hook):
        outputs = run_function(*args)
    return outputs


def CheckpointRecomputeFunction(run_function: Callable, *args):
    return torch.utils.checkpoint.checkpoint(run_function, *args, use_reentrant=True)
