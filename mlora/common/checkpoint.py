from typing import Callable, Tuple

import torch


def pack_hook(to_offload: torch.Tensor) -> Tuple[torch.device, torch.Tensor]:
    return to_offload.device, to_offload.to("cpu")


def unpack_hook(to_offload_info: Tuple[torch.device, torch.Tensor]) -> torch.Tensor:
    device, to_offload = to_offload_info
    return to_offload.to(device)


def CheckpointNoneFunction(run_function: Callable, *args):
    return run_function(*args)


def CheckpointOffloadFunction(run_function: Callable, *args):
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        outputs = run_function(*args)
    return outputs


def CheckpointRecomputeFunction(run_function: Callable, *args):
    return torch.utils.checkpoint.checkpoint(run_function, *args, use_reentrant=True)


CHECKPOINT_CLASSES = {
    "none": CheckpointNoneFunction,
    "offload": CheckpointOffloadFunction,
    "recompute": CheckpointRecomputeFunction,
}
