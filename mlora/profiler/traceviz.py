from typing import Set, Tuple

import torch
from graphviz import Digraph

G_NODE_ATTR = dict(
    style="filled",
    shape="box",
    align="left",
    fontsize="10",
    ranksep="0.1",
    height="0.2",
    fontname="monospace",
)


def __sizeof_fmt(num, suffix="B"):
    for unit in ("", "K", "M", "G"):
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} T{suffix}"


def __name_of_size(var: torch.Tensor):
    size_arr = ["%d" % v for v in var.size()]
    memory_size = var.element_size() * var.nelement()
    return (
        f'[{", ".join(size_arr)}] * {str(var.element_size())} '
        + f"= {__sizeof_fmt(memory_size / 8)}"
    )


def __name_of_grad_fn(var: torch.autograd.graph.Node) -> str:
    class_name = var.name()
    split_index = class_name.rfind("::")
    if split_index != -1:
        class_name = class_name[split_index + 2 :]
    return class_name


def __add_the_attr_to_name(grad_name: str, attr: str, var: torch.Tensor) -> str:
    grad_name += "\n"
    grad_name += attr + " : " + __name_of_size(var)
    return grad_name


def __tuple_tensor_add_attr_to_name(grad_fn_name: str, attr: str, var: Tuple) -> str:
    for item_val in var:
        grad_fn_name = __add_the_attr_to_name(grad_fn_name, attr, item_val)
    return grad_fn_name


def __dot_add_nodes(dot: Digraph, grad_fn: torch.autograd.graph.Node, visited: set):
    assert isinstance(grad_fn, torch.autograd.graph.Node)
    if grad_fn in visited:
        return
    visited.add(grad_fn)

    grad_fn_name = __name_of_grad_fn(grad_fn)

    for attr in dir(grad_fn):
        if not attr.startswith("_saved"):
            continue
        var: torch.Tensor | Tuple[torch.Tensor, ...] = getattr(grad_fn, attr)

        if isinstance(var, torch.Tensor):
            var_tensor: torch.Tensor = var
            grad_fn_name = __add_the_attr_to_name(grad_fn_name, attr, var_tensor)
        else:
            var_tuple: Tuple[torch.Tensor, ...] = var
            grad_fn_name = __tuple_tensor_add_attr_to_name(
                grad_fn_name, attr, var_tuple
            )

    if "__tp_name" in grad_fn.metadata():
        grad_fn_name += "\ntracepoint : " + grad_fn.metadata()["__tp_name"]

    dot.node(str(id(grad_fn)), grad_fn_name)

    if hasattr(grad_fn, "variable"):
        grad_var: torch.Tensor = grad_fn.variable
        dot.node(str(id(var)), __name_of_size(grad_var), fillcolor="lightblue")
        dot.edge(str(id(var)), str(id(grad_fn)))

    if hasattr(grad_fn, "saved_tensors"):
        for item_val in grad_fn.saved_tensors:
            dot.node(str(id(item_val)), __name_of_size(item_val), fillcolor="orange")
            dot.edge(str(id(item_val)), str(id(grad_fn)), dir="none")

    if hasattr(grad_fn, "next_functions"):
        for item_grad in grad_fn.next_functions:
            if item_grad[0] is not None:
                __dot_add_nodes(dot, item_grad[0], visited)
                dot.edge(str(id(item_grad[0])), str(id(grad_fn)))


def __add_base_tensor(dot: Digraph, var: torch.Tensor, visited: Set[torch.Tensor]):
    assert isinstance(var, torch.Tensor)
    if var in visited:
        return
    visited.add(var)
    dot.node(str(id(var)), __name_of_size(var), fillcolor="darkolivegreen1")

    if var.grad_fn:
        __dot_add_nodes(dot, var.grad_fn, visited)
        dot.edge(str(id(var.grad_fn)), str(id(var)))
    if var._base is not None:
        __add_base_tensor(dot, var._base, visited)
        dot.edge(str(id(var._base)), str(id(var)), style="dotted")


def trace(var: torch.Tensor, file_name: str):
    dot = Digraph(node_attr=G_NODE_ATTR, graph_attr=dict(size="12,12"))

    visited: Set[torch.Tensor] = set()
    __add_base_tensor(dot, var, visited)
    visited.clear()

    # resize graph
    num_rows = len(dot.body)
    content_size = num_rows * 0.15
    size = max(12, content_size)
    size_str = str(size) + "," + str(size)
    dot.graph_attr.update(size=size_str)

    dot.save(filename=file_name)
