import json

import requests
from InquirerPy import inquirer, separator, validator
from rich import print
from rich.box import ASCII
from rich.table import Table

from .setting import url

g_file_type_map = {"train data": "data", "prompt template": "prompt"}


def list_file(obj, file_type: str):
    ret = requests.get(url() + f"/{file_type}")
    ret_items = json.loads(ret.text)

    table = Table(show_header=True, show_lines=True, box=ASCII)
    table.add_column("name", justify="center")
    table.add_column("file", justify="center")

    for item in ret_items:
        row_data = [item["name"], item["file"]]
        table.add_row(*row_data)

    obj.ret_ = ret_items
    obj.pret_ = table


def upload_file():
    file_type = inquirer.select(
        message="file type:",
        choices=[separator.Separator(), *g_file_type_map.keys(), separator.Separator()],
    ).execute()
    file_type = g_file_type_map[file_type]

    name = inquirer.text(
        message="name:",
        validate=validator.EmptyInputValidator("name should not be empty"),
    ).execute()

    post_url = url() + f"/{file_type}?name={name}"

    path = inquirer.filepath(
        message="file path:",
        default="/",
        validate=validator.PathValidator(is_file=True, message="input is not a file"),
        only_files=True,
    ).execute()

    ret = requests.post(post_url, files={"data_file": open(path, "rb")})

    print(json.loads(ret.text))


def delete_file(obj):
    list_file(obj, "data")
    data_file_list = [("data", item["name"]) for item in obj.ret_]

    list_file(obj, "prompt")
    prompt_file_list = [("prompt", item["name"]) for item in obj.ret_]

    chose_item = inquirer.select(
        message="file name:",
        choices=[
            separator.Separator(),
            *data_file_list,
            *prompt_file_list,
            separator.Separator(),
        ],
    ).execute()

    delete_url = url() + f"/{chose_item[0]}?name={chose_item[1]}"

    ret = requests.delete(delete_url)

    print(json.loads(ret.text))


def help_file(_):
    print("Usage of file:")
    print("  ls")
    print("    list the train or prompt data.")
    print("  upload")
    print("    upload a train or prompt data.")
    print("  delete")
    print("    delete a train or prompt data.")


def do_file(obj, args):
    args = args.split(" ")

    if args[0] == "ls":
        # to chose file type
        list_file(obj, "data")
        print("Data files:")
        print(obj.pret_)

        list_file(obj, "prompt")
        print("Prompt files:")
        return print(obj.pret_)
    elif args[0] == "upload":
        return upload_file()
    elif args[0] == "delete":
        return delete_file(obj)

    help_file(None)
