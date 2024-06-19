import json
import requests
from InquirerPy import inquirer
from rich import print
from rich.table import Table
from rich.box import ASCII

from .setting import url
from .file import list_file


def list_dataset(obj):
    ret = requests.get(url() + "/dataset")
    ret = json.loads(ret.text)

    table = Table(show_header=True, show_lines=True, box=ASCII)
    table.add_column("name", justify="left")
    table.add_column("train", justify="center")
    table.add_column("prompt", justify="center")
    table.add_column("preprocess", justify="center")

    obj.ret_ = []

    for item in ret:
        item = json.loads(item)
        table.add_row(item["name"], item["train"],
                      item["prompt"], item["preprocess"])
        obj.ret_.append(item["name"])

    obj.pret_ = table


def create_dataset(obj):
    name = inquirer.text(
        message="name:").execute()

    list_file(obj, "train")
    all_train = [item["name"] for item in obj.ret_]
    if len(all_train) == 0:
        print("no train data, please upload one")
        return

    list_file(obj, "prompt")
    all_prompt = [item["name"] for item in obj.ret_]
    if len(all_prompt) == 0:
        print("no prompt template file, please upload one")
        return

    use_train = inquirer.select(
        message="train data file:", choices=all_train).execute()
    use_prompt = inquirer.select(
        message="prompt template file:", choices=all_prompt).execute()
    use_preprocess = inquirer.select(
        message="data preprocessing:", choices=["default", "shuffle", "sort"]).execute()

    ret = requests.post(url() + "/dataset", json={
        "name": name,
        "train": use_train,
        "prompt": use_prompt,
        "preprocess": use_preprocess
    })

    print(json.loads(ret.text))


def help_dataset(_):
    print("Usage of dataset:")
    print("  ls")
    print("    list all the dataset.")
    print("  create")
    print("    create a new dataset.")


def do_dataset(obj, args):
    args = args.split(" ")

    if args[0] == "ls":
        list_dataset(obj)
        return print(obj.pret_)
    elif args[0] == "create":
        return create_dataset(obj)

    help_dataset(None)
