from mlora.checkpoint.checkpoint import (check_backward_validity,
                                         _get_autocast_kwargs,
                                         get_device_states,
                                         set_device_states,
                                         detach_variable)
from mlora.profiler.profiler import tensors_nvtx_wrapper_by_tracepoint

import torch


class CheckpointRecomputeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.gpu_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs()
        ctx.fwd_cpu_state = torch.get_rng_state()

        ctx.had_cuda_in_fwd = False
        if torch.cuda._initialized:
            ctx.had_cuda_in_fwd = True
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
        if ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices
        with torch.random.fork_rng(devices=rng_devices):
            torch.set_rng_state(ctx.fwd_cpu_state)
            if ctx.had_cuda_in_fwd:
                set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
            detached_inputs = detach_variable(tuple(inputs))
            with torch.enable_grad(), \
                    torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                    torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
                outputs = ctx.run_function(*detached_inputs)
                if isinstance(outputs, torch.Tensor):
                    outputs = (outputs,)
                # only in enable grad context can wrapper the tracepoint
                tensors_nvtx_wrapper_by_tracepoint(outputs)

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
