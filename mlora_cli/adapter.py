import json
from typing import Any, Dict

import requests
from InquirerPy import inquirer, separator, validator
from InquirerPy.base import Choice
from rich import print
from rich.box import ASCII
from rich.table import Table

from .setting import url


def list_adapter(obj):
    ret = requests.get(url() + "/adapter")
    ret_items = json.loads(ret.text)

    table = Table(show_header=True, show_lines=True, box=ASCII)
    table.add_column("name", justify="center")
    table.add_column("type", justify="center")
    table.add_column("dir", justify="center")
    table.add_column("state", justify="center")
    table.add_column("task", justify="center")

    obj.ret_ = []

    for ret_item in ret_items:
        item = json.loads(ret_item)
        table.add_row(
            item["name"], item["type"], item["path"], item["state"], item["task"]
        )
        obj.ret_.append((item["name"], item["state"], item["task"]))

    obj.pret_ = table


def adapter_type_set(adapter_conf: Dict[str, Any]):
    adapter_type = inquirer.select(
        message="type:",
        choices=[separator.Separator(), "lora", "loraplus", separator.Separator()],
    ).execute()
    adapter_conf["type"] = adapter_type

    if adapter_type == "loraplus":
        lr_ratio = inquirer.number(
            message="lr_ratio:", float_allowed=True, default=8.0, replace_mode=True
        ).execute()
        adapter_conf["lr_ratio"] = lr_ratio

    return adapter_conf


def adapter_optimizer_set(adapter_conf: Dict[str, Any]):
    optimizer = inquirer.select(
        message="optimizer:",
        choices=[separator.Separator(), "adamw", "sgd", separator.Separator()],
    ).execute()
    adapter_conf["optimizer"] = optimizer

    lr = inquirer.number(
        message="learning rate:", float_allowed=True, default=3e-4, replace_mode=True
    ).execute()
    adapter_conf["lr"] = lr

    if optimizer == "sgd":
        momentum = inquirer.number(
            message="momentum:", float_allowed=True, default=0.0, replace_mode=True
        ).execute()
        adapter_conf["momentum"] = momentum
    return adapter_conf


def adapter_lr_scheduler_set(adapter_conf: Dict[str, Any]):
    need_lr_scheduler = inquirer.confirm(
        message="Need learning rate scheduler:", default=False
    ).execute()
    if not need_lr_scheduler:
        return adapter_conf

    lr_scheduler_type = inquirer.select(
        message="optimizer:",
        choices=[separator.Separator(), "cosine", separator.Separator()],
    ).execute()
    adapter_conf["lrscheduler"] = lr_scheduler_type

    if lr_scheduler_type == "cosine":
        t_max = inquirer.number(
            message="maximum number of iterations:",
            replace_mode=True,
            default=100000,
        ).execute()
        adapter_conf["t_max"] = t_max

        eta_min = inquirer.number(
            message="minimum learning rate:",
            float_allowed=True,
            replace_mode=True,
            default=0.0,
        ).execute()
        adapter_conf["eta_min"] = eta_min

    return adapter_conf


def adapter_set(adapter_conf: Dict[str, Any]):
    r = inquirer.number(message="rank:", default=32).execute()
    adapter_conf["r"] = r

    alpha = inquirer.number(message="alpha:", default=64).execute()
    adapter_conf["alpha"] = alpha

    dropout = inquirer.number(
        message="dropout:", float_allowed=True, replace_mode=True, default=0.05
    ).execute()
    adapter_conf["dropout"] = dropout

    adapter_conf["target_modules"] = {
        "q_proj": False,
        "k_proj": False,
        "v_proj": False,
        "o_proj": False,
        "gate_proj": False,
        "down_proj": False,
        "up_proj": False,
    }
    target_modules = inquirer.checkbox(
        message="target_modules:",
        choices=[
            separator.Separator(),
            Choice("q_proj", enabled=True),
            Choice("k_proj", enabled=True),
            Choice("v_proj", enabled=True),
            Choice("o_proj", enabled=True),
            Choice("gate_proj", enabled=False),
            Choice("down_proj", enabled=False),
            Choice("up_proj", enabled=False),
            separator.Separator(),
        ],
    ).execute()
    for target in target_modules:
        adapter_conf["target_modules"][target] = True

    return adapter_conf


def create_adapter():
    adapter_conf = {}

    name = inquirer.text(
        message="name:",
        validate=validator.EmptyInputValidator("Input should not be empty"),
    ).execute()
    adapter_conf["name"] = name

    adapter_conf = adapter_type_set(adapter_conf)
    adapter_conf = adapter_optimizer_set(adapter_conf)
    adapter_conf = adapter_lr_scheduler_set(adapter_conf)
    adapter_conf = adapter_set(adapter_conf)

    ret = requests.post(url() + "/adapter", json=adapter_conf)

    print(json.loads(ret.text))


def delete_adapter(obj):
    list_adapter(obj)
    all_adapters = obj.ret_
    all_adapters = [
        item for item in all_adapters if item[2] == "NO" or item[1] == "DONE"
    ]

    if len(all_adapters) == 0:
        print("no adapter, please create one")
        return

    adapter_name = inquirer.select(
        message="adapter name:",
        choices=[separator.Separator(), *all_adapters, separator.Separator()],
    ).execute()

    ret = requests.delete(url() + f"/adapter?name={adapter_name[0]}")
    ret = json.loads(ret.text)

    print(ret)


def help_adapter(_):
    print("Usage of adapter:")
    print("  ls")
    print("    list all the adapter.")
    print("  create")
    print("    create a new adapter.")
    print("  delete")
    print("    delete a adapter.")


def do_adapter(obj, args):
    args = args.split(" ")

    if args[0] == "ls":
        list_adapter(obj)
        return print(obj.pret_)
    elif args[0] == "create":
        return create_adapter()
    elif args[0] == "delete":
        return delete_adapter(obj)

    help_adapter(None)
