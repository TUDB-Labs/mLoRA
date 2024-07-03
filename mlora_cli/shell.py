import cmd

from .adapter import do_adapter, help_adapter
from .dataset import do_dataset, help_dataset
from .dispatcher import do_dispatcher, help_dispatcher
from .file import do_file, help_file
from .setting import do_set, help_set
from .task import do_task, help_task


def help_quit(_):
    print("Quit the cli")


def do_quit(*_):
    exit(0)


class mLoRAShell(cmd.Cmd):
    intro = "Welcome to the mLoRA CLI. Type help or ? to list commands.\n"
    prompt = "(mLoRA) "

    help_quit = help_quit
    do_quit = do_quit

    help_dispatcher = help_dispatcher
    do_dispatcher = do_dispatcher

    help_file = help_file
    do_file = do_file

    help_dataset = help_dataset
    do_dataset = do_dataset

    help_adapter = help_adapter
    do_adapter = do_adapter

    help_task = help_task
    do_task = do_task

    help_set = help_set
    do_set = do_set


def cmd_loop():
    mLoRAShell().cmdloop()
