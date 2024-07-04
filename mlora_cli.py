import cmd

import mlora.cli


def help_quit(_):
    print("Quit the cli")


def do_quit(*_):
    exit(0)


class mLoRAShell(cmd.Cmd):
    intro = 'Welcome to the mLoRA CLI. Type help or ? to list commands.\n'
    prompt = '(mLoRA) '

    help_quit = help_quit
    do_quit = do_quit

    help_dispatcher = mlora.cli.help_dispatcher
    do_dispatcher = mlora.cli.do_dispatcher

    help_file = mlora.cli.help_file
    do_file = mlora.cli.do_file

    help_dataset = mlora.cli.help_dataset
    do_dataset = mlora.cli.do_dataset

    help_adapter = mlora.cli.help_adapter
    do_adapter = mlora.cli.do_adapter

    help_task = mlora.cli.help_task
    do_task = mlora.cli.do_task


if __name__ == '__main__':
    mLoRAShell().cmdloop()
