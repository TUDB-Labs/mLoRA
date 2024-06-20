import json
import requests
from InquirerPy import inquirer
from rich import print
from rich.table import Table
from rich.box import ASCII

from .adapter import list_adapter
from .dataset import list_dataset
from .setting import url


def list_task(obj):
    ret = requests.get(url() + "/task")
    ret = json.loads(ret.text)

    table = Table(show_header=True, show_lines=True, box=ASCII)
    table.add_column("name", justify="center")
    table.add_column("type", justify="center")
    table.add_column("dataset", justify="center")
    table.add_column("adapter", justify="center")
    table.add_column("state", justify="center")

    for item in ret:
        item = json.loads(item)
        table.add_row(item["name"], item["type"],
                      item["dataset"], item["adapter"], item["state"])

    obj.pret_ = table


def task_type_set(task_conf, all_adapters):
    if task_conf["type"] == "dpo" or task_conf["type"] == "cpo":
        beta = inquirer.number(
            message="beta:",
            float_allowed=True,
            default=0.1,
            replace_mode=True
        ).execute()
        task_conf["beta"] = beta

        label_smoothing = inquirer.number(
            message="label_smoothing:",
            float_allowed=True,
            default=0.0,
            replace_mode=True
        ).execute()
        task_conf["label_smoothing"] = label_smoothing

    if task_conf["type"] == "cpo":
        loss_type = inquirer.select(
            message="loss_type:", choices=["sigmoid", "hinge"]).execute()
        task_conf["loss_type"] = loss_type

    if task_conf["type"] == "dpo":
        loss_type = inquirer.select(
            message="loss_type:", choices=["sigmoid", "ipo"]).execute()
        task_conf["loss_type"] = loss_type

        all_adapters.append("base")
        reference = inquirer.select(
            message="reference model:", choices=all_adapters).execute()
        task_conf["reference"] = reference

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
        message="type:", choices=["train", "dpo", "cpo"]).execute()
    task_conf["type"] = task_type

    name = inquirer.text(
        message="name:").execute()
    task_conf["name"] = name

    list_dataset(obj)
    all_dataset = obj.ret_

    if len(all_dataset) == 0:
        print("no dataset, please create one")
        return

    dataset = inquirer.select(
        message="dataset:", choices=all_dataset).execute()
    task_conf["dataset"] = dataset

    list_adapter(obj)
    all_adapters = obj.ret_

    if len(all_adapters) == 0:
        print("no adapter can be train, please create one")
        return

    adapter = inquirer.select(
        message="train adapter:", choices=all_adapters).execute()
    task_conf["adapter"] = adapter

    task_conf = task_type_set(task_conf, all_adapters.copy())
    task_conf = task_set(task_conf)

    ret = requests.post(url() + "/task", json=task_conf)

    print(json.loads(ret.text))


def help_task(_):
    print("Usage of task:")
    print("  ls")
    print("    list the task.")
    print("  create")
    print("    create a task.")


def do_task(obj, args):
    args = args.split(" ")

    if args[0] == "ls":
        list_task(obj)
        return print(obj.pret_)
    elif args[0] == "create":
        return create_task(obj)

    help_task(None)
