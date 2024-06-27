import json

import requests
from InquirerPy import inquirer, separator, validator
from rich import print
from rich.box import ASCII
from rich.table import Table

from .setting import url

g_file_type_map = {"train data": "data", "prompt data": "prompt"}


def list_file(obj, file_type: str):
    ret = requests.get(url() + f"/{file_type}")
    ret_items = json.loads(ret.text)

    table = Table(show_header=True, show_lines=True, box=ASCII)
    table.add_column("name", justify="center")
    table.add_column("file", justify="center")
    if file_type == "prompt":
        table.add_column("prompter", justify="center")

    for item in ret_items:
        row_data = [item["name"], item["file"]["file_path"]]
        if file_type == "prompt":
            row_data.append(item["file"]["prompt_type"])
        table.add_row(*row_data)

    obj.ret_ = ret_items
    obj.pret_ = table


def upload_file():
    name = inquirer.text(
        message="name:",
        validate=validator.EmptyInputValidator("name should not be empty"),
    ).execute()

    file_type = inquirer.select(
        message="file type:",
        choices=[separator.Separator(), *g_file_type_map.keys(), separator.Separator()],
    ).execute()
    file_type = g_file_type_map[file_type]

    post_url = url() + f"/{file_type}?name={name}"

    if file_type == "prompt":
        prompt_type = inquirer.select(
            message="prompter type:",
            choices=[
                separator.Separator(),
                "instruction",
                "preference",
                separator.Separator(),
            ],
        ).execute()
        post_url += f"&prompt_type={prompt_type}"

    path = inquirer.filepath(
        message="file path:",
        default="/",
        validate=validator.PathValidator(is_file=True, message="input is not a file"),
        only_files=True,
    ).execute()

    ret = requests.post(post_url, files={"data_file": open(path, "rb")})

    print(json.loads(ret.text))


def help_file(_):
    print("Usage of file:")
    print("  ls")
    print("    list the usable data or prompt data.")
    print("  upload")
    print("    upload a training data or prompt data.")


def do_file(obj, args):
    args = args.split(" ")

    if args[0] == "ls":
        # to chose file type
        file_type = inquirer.select(
            message="type:",
            choices=[
                separator.Separator(),
                *g_file_type_map.keys(),
                separator.Separator(),
            ],
        ).execute()
        file_type = g_file_type_map[file_type]
        list_file(obj, file_type)
        return print(obj.pret_)
    elif args[0] == "upload":
        return upload_file()

    help_file(None)
