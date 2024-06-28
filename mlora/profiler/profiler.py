import logging
from contextlib import contextmanager
from typing import Callable, List, Set, Tuple

import torch

TRACEPOINT_KEY = "__tp_name"


g_is_trace_model = False


def setup_trace_mode():
    global g_is_trace_model
    logging.info("m-LoRA setup to the trace mode.")
    g_is_trace_model = True


def is_trace_model() -> bool:
    global g_is_trace_model
    return g_is_trace_model


def __get_scope_name(grad_fn: torch.autograd.graph.Node):
    if TRACEPOINT_KEY in grad_fn.metadata():
        return grad_fn.metadata()[TRACEPOINT_KEY]
    return grad_fn.name()


def nvtx_range_wrapper(func: Callable, msg: str):
    if not is_trace_model():
        return func

    def wrap(*args, **kwargs):
        with torch.cuda.nvtx.range(msg=msg):
            return func(*args, **kwargs)

    return wrap


def nvtx_wrapper(msg: str):
    def func_decorator(func):
        return nvtx_range_wrapper(func, msg)

    return func_decorator


@contextmanager
def nvtx_range(msg, *args, **kwargs):
    if not is_trace_model():
        yield
        return
    torch.cuda.nvtx.range_push(msg.format(*args, **kwargs))
    yield
    torch.cuda.nvtx.range_pop()


g_scope_stack: List[str] = []


def __nvtx_pre_hook_wrapper(func: Callable, grad_fn: torch.autograd.graph.Node):
    global g_scope_stack

    scope_name = __get_scope_name(grad_fn)

    def wrap(*args, **kwargs):
        if len(g_scope_stack) == 0:
            g_scope_stack.append(scope_name)
            torch.cuda.nvtx.range_push(scope_name)
        elif g_scope_stack[-1] != scope_name:
            g_scope_stack.pop()
            torch.cuda.nvtx.range_pop()

            g_scope_stack.append(scope_name)
            torch.cuda.nvtx.range_push(scope_name)
        else:
            # just for pretty
            pass

        return func(*args, **kwargs)

    return wrap


def __nvtx_hook_wrapper(func: Callable, grad_fn: torch.autograd.graph.Node):
    global g_scope_stack

    # do not capture the func object, will cost memory to hold it
    is_last_node = (
        not hasattr(grad_fn, "next_functions") or len(grad_fn.next_functions) == 0
    )

    def wrap(*args, **kwargs):

        if is_last_node and len(g_scope_stack) > 0:
            g_scope_stack.pop()
            torch.cuda.nvtx.range_pop()

        return func(*args, **kwargs)

    return wrap


def __grad_fn_pre_hook_dummy(grad_outputs: Tuple[torch.Tensor]) -> None:
    return None


def __grad_fn_hook_dummy(
    grad_inputs: Tuple[torch.Tensor], grad_outputs: Tuple[torch.Tensor]
) -> None:
    return None


def __grad_fn_nvtx_wrapper(grad_fn: torch.autograd.graph.Node):
    if not torch.is_grad_enabled():
        return

    assert isinstance(
        grad_fn, torch.autograd.graph.Node
    ), f"error type: {type(grad_fn)}"

    grad_fn.register_prehook(__nvtx_pre_hook_wrapper(__grad_fn_pre_hook_dummy, grad_fn))
    grad_fn.register_hook(__nvtx_hook_wrapper(__grad_fn_hook_dummy, grad_fn))


def set_backward_tracepoint(
    grad_fn: torch.autograd.graph.Node | None, tp_name: str, recursion: bool = True
):
    if not is_trace_model():
        return
    # tp - tracepoint
    if not torch.is_grad_enabled():
        return

    assert isinstance(
        grad_fn, torch.autograd.graph.Node
    ), f"error type: {type(grad_fn)}"

    if TRACEPOINT_KEY in grad_fn.metadata():
        return

    if not recursion:
        grad_fn.metadata()[TRACEPOINT_KEY] = tp_name
        return

    visited: Set[torch.autograd.graph.Node] = set()
    to_visited_stack: List[torch.autograd.graph.Node] = []

    to_visited_stack.append(grad_fn)

    while len(to_visited_stack) > 0:
        to_visit = to_visited_stack.pop()
        to_visit.metadata()[TRACEPOINT_KEY] = tp_name

        visited.add(to_visit)

        if not hasattr(to_visit, "next_functions") or len(to_visit.next_functions) == 0:
            continue

        for next_fn in reversed(to_visit.next_functions):
            if next_fn[0] is None:
                continue
            if next_fn[0] in visited:
                continue
            if TRACEPOINT_KEY in next_fn[0].metadata():
                continue
            to_visited_stack.append(next_fn[0])

    visited.clear()


def grad_fn_nvtx_wrapper_by_tracepoint(grad_fn: torch.autograd.graph.Node):
    if not is_trace_model():
        return

    visited: Set[torch.autograd.graph.Node] = set()
    to_visited_stack: List[torch.autograd.graph.Node] = []
    to_visited_stack.append(grad_fn)

    while len(to_visited_stack) > 0:
        to_visit = to_visited_stack.pop()
        __grad_fn_nvtx_wrapper(to_visit)

        visited.add(to_visit)

        if not hasattr(to_visit, "next_functions") or len(to_visit.next_functions) == 0:
            continue

        for next_fn in reversed(to_visit.next_functions):
            if next_fn[0] is None:
                continue
            if next_fn[0] in visited:
                continue
            to_visited_stack.append(next_fn[0])

    visited.clear()
