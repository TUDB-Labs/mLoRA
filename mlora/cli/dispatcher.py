import json

import requests
from rich import print
from rich.box import ASCII
from rich.table import Table

from .setting import url


def help_dispatcher(_):
    print("List the configuration about dispatcher.")


def do_dispatcher(*_):
    ret = requests.get(url() + "/dispatcher")
    ret_text = json.loads(ret.text)

    table = Table(show_header=True, show_lines=True, box=ASCII)
    table.add_column("Item", justify="center")
    table.add_column("Value", justify="center")
    for item, value in ret_text.items():
        table.add_row(item, str(value))

    print(table)
