from typing import Any, Iterable, List, Tuple
from mlora.backends import _backend

import torch


def detach_variable(inputs: Tuple[Any, ...]) -> Tuple[torch.Tensor, ...]:
    if isinstance(inputs, tuple):
        out = []
        for inp in inputs:
            if not isinstance(inp, torch.Tensor):
                out.append(inp)
                continue

            x = inp.detach()
            x.requires_grad = inp.requires_grad
            out.append(x)
        return tuple(out)
    else:
        raise RuntimeError("Only tuple tensors is supported.")


def check_backward_validity(inputs: Iterable[Any]) -> None:
    if not any(inp.requires_grad for inp in inputs if isinstance(inp, torch.Tensor)):
        raise RuntimeError("Input need grad")


def get_device_states(*args) -> Tuple[List[int], List[torch.Tensor]]:
    fwd_gpu_devices = list({arg.device.index for arg in args
                            if isinstance(arg, torch.Tensor) and arg.device.type == _backend.device_name()})

    fwd_gpu_states = []
    for device in fwd_gpu_devices:
        fwd_gpu_states.append(_backend.get_rng_state(device))

    return fwd_gpu_devices, fwd_gpu_states


def set_device_states(devices, states) -> None:
    for device, state in zip(devices, states):
        _backend.set_rng_state(device, state)


def _get_autocast_kwargs():
    gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                           "dtype": torch.get_autocast_gpu_dtype(),
                           "cache_enabled": torch.is_autocast_cache_enabled()}

    cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                           "dtype": torch.get_autocast_cpu_dtype(),
                           "cache_enabled": torch.is_autocast_cache_enabled()}

    return gpu_autocast_kwargs, cpu_autocast_kwargs


def pack_hook(to_offload: torch.Tensor) -> Tuple[torch.device, torch.Tensor]:
    return to_offload.device, to_offload.to("cpu")


def unpack_hook(to_offload_info: Tuple[torch.device, torch.Tensor]) -> torch.Tensor:
    device, to_offload = to_offload_info
    return to_offload.to(device)


def CheckpointOffloadFunction(run_function, *args):
    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        outputs = run_function(*args)
    return outputs


class CheckpointRecomputeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.gpu_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs()
        ctx.fwd_cpu_state = torch.get_rng_state()

        ctx.had_gpu_in_fwd = False
        if _backend.is_initialized():
            ctx.had_gpu_in_fwd = True
            ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(
                *args)

        ctx.inputs = []
        ctx.tensor_indices = []
        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                tensor_inputs.append(arg)
                ctx.tensor_indices.append(i)
                ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)

        ctx.save_for_backward(*tensor_inputs)

        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("Checkpoint invalid")
        inputs = list(ctx.inputs)
        tensor_indices = ctx.tensor_indices
        tensors = ctx.saved_tensors

        for i, idx in enumerate(tensor_indices):
            inputs[idx] = tensors[i]

        rng_devices = []
        if ctx.had_gpu_in_fwd:
            rng_devices = ctx.fwd_gpu_devices
        with _backend.fork_rng(rng_devices):
            torch.set_rng_state(ctx.fwd_cpu_state)
            if ctx.had_gpu_in_fwd:
                set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
            detached_inputs = detach_variable(tuple(inputs))
            with torch.enable_grad(), _backend.autocast(**ctx.gpu_autocast_kwargs), \
                    torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
                outputs = ctx.run_function(*detached_inputs)
                if isinstance(outputs, torch.Tensor):
                    outputs = (outputs,)

        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError("No output with grad")
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(inp.grad if isinstance(inp, torch.Tensor) else None
                      for inp in detached_inputs)

        return (None,) + grads
