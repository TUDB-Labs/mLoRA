import json
import requests
from InquirerPy import inquirer, separator
from rich import print
from rich.table import Table
from rich.box import ASCII

from .setting import url
from .file import list_file


def list_dataset(obj):
    ret = requests.get(url() + "/dataset")
    ret = json.loads(ret.text)

    table = Table(show_header=True, show_lines=True, box=ASCII)
    table.add_column("name", justify="center")
    table.add_column("train data name", justify="center")
    table.add_column("prompt data name", justify="center")
    table.add_column("prompter", justify="center")
    table.add_column("preprocess", justify="center")

    obj.ret_ = []

    for item in ret:
        item = json.loads(item)
        table.add_row(item["name"],
                      item["data_name"],
                      item["prompt_name"],
                      item["prompt_type"],
                      item["preprocess"])
        obj.ret_.append(item["name"])

    obj.pret_ = table


def create_dataset(obj):
    name = inquirer.text(
        message="name:").execute()

    list_file(obj, "data")
    all_train_data = [item["name"] for item in obj.ret_]
    if len(all_train_data) == 0:
        print("no train data, please upload one")
        return

    list_file(obj, "prompt")
    all_prompt = [item["name"] for item in obj.ret_]
    if len(all_prompt) == 0:
        print("no prompt template file, please upload one")
        return

    use_train = inquirer.select(
        message="train data file:", choices=[separator.Separator(),
                                             *all_train_data,
                                             separator.Separator()]).execute()
    use_prompt = inquirer.select(
        message="prompt template file:", choices=[separator.Separator(),
                                                  *all_prompt,
                                                  separator.Separator()]).execute()
    use_preprocess = inquirer.select(
        message="data preprocessing:", choices=[separator.Separator(),
                                                "default",
                                                "shuffle",
                                                "sort",
                                                separator.Separator()]).execute()

    ret = requests.post(url() + "/dataset", json={
        "name": name,
        "data_name": use_train,
        "prompt_name": use_prompt,
        "preprocess": use_preprocess
    })

    print(json.loads(ret.text))


def showcase_dataset(obj):
    list_dataset(obj)
    all_dataset = obj.ret_

    if len(all_dataset) == 0:
        print("no dataset, please create one")
        return

    use_dataset = inquirer.select(
        message="dataset name:", choices=[separator.Separator(),
                                          *all_dataset,
                                          separator.Separator()]).execute()

    ret = requests.get(url() + f"/showcase?name={use_dataset}")
    ret = json.loads(ret.text)

    print(ret)


def help_dataset(_):
    print("Usage of dataset:")
    print("  ls")
    print("    list all the dataset.")
    print("  create")
    print("    create a new dataset.")
    print("  showcase")
    print("    display training data composed of prompt and dataset.")


def do_dataset(obj, args):
    args = args.split(" ")

    if args[0] == "ls":
        list_dataset(obj)
        return print(obj.pret_)
    elif args[0] == "create":
        return create_dataset(obj)
    elif args[0] == "showcase":
        return showcase_dataset(obj)

    help_dataset(None)
