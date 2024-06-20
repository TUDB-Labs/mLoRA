import json
import requests
from InquirerPy import inquirer, validator
from rich import print
from rich.table import Table
from rich.box import ASCII


from .setting import url


def list_file(obj, file_type: str):
    ret = requests.get(url() + f"/{file_type}")
    ret = json.loads(ret.text)

    table = Table(show_header=True, show_lines=True, box=ASCII)
    table.add_column("name", justify="center")
    table.add_column("file", justify="center")

    for item in ret:
        table.add_row(item["name"], item["file"])

    obj.ret_ = ret
    obj.pret_ = table


def upload_file():
    file_type = inquirer.select(
        message="type:", choices=["train", "prompt"]).execute()
    name = inquirer.text(
        message="name:",
        validate=validator.EmptyInputValidator("name should not be empty")).execute()
    path = inquirer.filepath(
        message="file path:",
        default="/",
        validate=validator.PathValidator(
            is_file=True, message="input is not a file"),
        only_files=True).execute()

    ret = requests.post(
        url() + f"/{file_type}?name={name}", files={"data_file": open(path, "rb")})

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
            message="type:", choices=["train", "prompt"]).execute()
        list_file(obj, file_type)
        return print(obj.pret_)
    elif args[0] == "upload":
        return upload_file()

    help_file(None)
