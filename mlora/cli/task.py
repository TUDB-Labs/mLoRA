import json

import requests
from InquirerPy import inquirer, separator, validator
from rich import print
from rich.box import ASCII
from rich.table import Table

from .adapter import list_adapter
from .dataset import list_dataset
from .setting import url


def list_task(obj):
    ret = requests.get(url() + "/task")
    ret_items = json.loads(ret.text)

    table = Table(show_header=True, show_lines=True, box=ASCII)
    table.add_column("name", justify="center")
    table.add_column("type", justify="center")
    table.add_column("dataset", justify="center")
    table.add_column("adapter", justify="center")
    table.add_column("state", justify="center")

    obj.ret_ = []
    for ret_item in ret_items:
        item = json.loads(ret_item)
        table.add_row(
            item["name"], item["type"], item["dataset"], item["adapter"], item["state"]
        )
        obj.ret_.append((item["name"], item["state"]))

    obj.pret_ = table


def task_type_set(obj, task_conf):
    if task_conf["type"] == "dpo" or task_conf["type"] == "cpo":
        beta = inquirer.number(
            message="beta:", float_allowed=True, default=0.1, replace_mode=True
        ).execute()
        task_conf["beta"] = beta

        label_smoothing = inquirer.number(
            message="label_smoothing:",
            float_allowed=True,
            default=0.0,
            replace_mode=True,
        ).execute()
        task_conf["label_smoothing"] = label_smoothing

    if task_conf["type"] == "cpo":
        loss_type = inquirer.select(
            message="loss_type:",
            choices=[separator.Separator(), "sigmoid", "hinge", separator.Separator()],
        ).execute()
        task_conf["loss_type"] = loss_type

    if task_conf["type"] == "dpo":
        loss_type = inquirer.select(
            message="loss_type:",
            choices=[separator.Separator(), "sigmoid", "ipo", separator.Separator()],
        ).execute()
        task_conf["loss_type"] = loss_type

        list_adapter(obj)
        all_ref_adapters = [
            item
            for item in obj.ret_
            if item[1] == "DONE" and item[0] != task_conf["adapter"]
        ]
        all_ref_adapters.append(("base", "use the base llm model"))
        reference = inquirer.select(
            message="reference model:",
            choices=[separator.Separator(), *all_ref_adapters, separator.Separator()],
        ).execute()
        task_conf["reference"] = reference[0]

    return task_conf


def task_set(task_conf):
    batch_size = inquirer.number(
        message="batch size:",
        replace_mode=True,
        default=16,
    ).execute()
    task_conf["batch_size"] = batch_size

    mini_batch_size = inquirer.number(
        message="mini batch size:",
        replace_mode=True,
        default=16,
    ).execute()
    task_conf["mini_batch_size"] = mini_batch_size

    num_epochs = inquirer.number(
        message="epochs:",
        replace_mode=True,
        default=10,
    ).execute()
    task_conf["num_epochs"] = num_epochs

    cutoff_len = inquirer.number(
        message="cutoff len:",
        replace_mode=True,
        default=256,
    ).execute()
    task_conf["cutoff_len"] = cutoff_len

    save_step = inquirer.number(
        message="save step:",
        replace_mode=True,
        default=100000,
    ).execute()
    task_conf["save_step"] = save_step

    return task_conf


def create_task(obj):
    task_conf = {}

    task_type = inquirer.select(
        message="type:",
        choices=[separator.Separator(), "train", "dpo", "cpo", separator.Separator()],
    ).execute()
    task_conf["type"] = task_type

    name = inquirer.text(
        message="name:",
        validate=validator.EmptyInputValidator("Input should not be empty"),
    ).execute()
    task_conf["name"] = name

    list_dataset(obj)
    all_dataset = obj.ret_

    if len(all_dataset) == 0:
        print("no dataset, please create one")
        return

    dataset = inquirer.select(
        message="dataset:",
        choices=[separator.Separator(), *all_dataset, separator.Separator()],
    ).execute()
    task_conf["dataset"] = dataset

    list_adapter(obj)
    all_adapters = obj.ret_

    if len(all_adapters) == 0:
        print("no adapter can be train, please create one")
        return

    adapter = inquirer.select(
        message="train adapter:",
        choices=[separator.Separator(), *all_adapters, separator.Separator()],
    ).execute()
    task_conf["adapter"] = adapter[0]

    task_conf = task_type_set(obj, task_conf)
    task_conf = task_set(task_conf)

    ret = requests.post(url() + "/task", json=task_conf)

    print(json.loads(ret.text))


def delete_task(obj):
    list_task(obj)
    all_task = obj.ret_

    delete_task = inquirer.select(
        message="termiate task:",
        choices=[separator.Separator(), *all_task, separator.Separator()],
    ).execute()

    delete_task_name = delete_task[0]

    ret = requests.delete(url() + f"/task?name={delete_task_name}")

    print(json.loads(ret.text))


def help_task(_):
    print("Usage of task:")
    print("  ls")
    print("    list the task.")
    print("  create")
    print("    create a task.")
    print("  delete")
    print("    delete a task.")


def do_task(obj, args):
    args = args.split(" ")

    if args[0] == "ls":
        list_task(obj)
        return print(obj.pret_)
    elif args[0] == "create":
        return create_task(obj)
    elif args[0] == "delete":
        return delete_task(obj)

    help_task(None)
