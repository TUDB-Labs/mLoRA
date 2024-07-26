#!/usr/bin/env python3
import json
import os

import fire

file_path = "templates"
work_path = os.path.dirname(os.path.abspath(__file__))


def compose_command(
    base_model: str,
    config: str = "mlora.json",
    inference: bool = False,
    evaluate: bool = False,
    load_adapter: bool = False,
    random_seed: int = 42,
    cuda_device: int = None,
    log_file: str = "mlora.log",
    overwrite: bool = False,
    attn_impl: str = None,
    sliding_window: bool = False,
    use_cache: bool = True,
    quantize: str = None,
    dtype: str = "bf16",
    tf32: bool = False,
):
    assert quantize in (None, "4bit", "8bit")
    assert dtype in ("fp32", "fp16", "bf16")
    command = "python mlora.py"
    if cuda_device is not None:
        command = f"CUDA_VISIBLE_DEVICES={cuda_device} " + command
    command += f" --base_model {base_model}"
    command += f" --config {config}"
    if inference:
        command += " --inference"
    if evaluate:
        command += " --evaluate"
    if load_adapter:
        command += " --load_adapter"
    command += f" --seed {random_seed}"
    command += f" --log_file {log_file}"
    if overwrite:
        command += " --overwrite"
    if attn_impl is not None:
        command += f" --attn_impl {attn_impl}"
    if sliding_window:
        command += " --sliding_window"
    if not use_cache:
        command += " --disable_cache"
    if quantize is not None:
        command += f" --load_{quantize}"
    if dtype in ("fp16", "bf16"):
        command += f" --{dtype}"
    if tf32:
        command += " --tf32"
    return os.system(command)


def update_record(dict_: dict, key_, value_):
    if value_ is not None:
        dict_[key_] = value_


def gen_config(
    # essential
    template: str,
    task_list: str,
    # optional
    adapter_name: str = None,
    file_name: str = "mlora.json",
    data_path: str = None,
    multi_task: bool = False,
    append: bool = False,
    # default value provided by template
    prompt_template: str = None,
    cutoff_len: int = None,
    save_step: int = None,
    lr_scheduler: str = None,
    warmup_steps: float = None,
    learning_rate: float = None,
    batch_size: int = None,
    micro_batch_size: int = None,
    evaluate_steps: int = None,
    evaluate_batch_size: int = None,
    num_epochs: int = None,
    loraplus_lr_ratio: float = None,
    use_dora: bool = None,
    use_rslora: bool = None,
    group_by_length: bool = None,
):
    import mlora

    template = f"{work_path}{os.sep}{file_path}{os.sep}{template}.json"
    config_dir = f"{work_path}{os.sep}{file_name}"

    with open(template, "r", encoding="utf8") as fp:
        template_obj = json.load(fp)

    update_record(template_obj, "cutoff_len", cutoff_len)
    update_record(template_obj, "save_step", save_step)
    lora_templates = template_obj["lora"]
    template_obj["lora"] = []

    if append:
        with open(config_dir, "r", encoding="utf8") as fp:
            orig_config = json.load(fp)
        template_obj["lora"] = orig_config["lora"]

    index = len(template_obj["lora"])
    if multi_task:
        task_list = [task_list]
        path_list = [data_path]
    else:
        task_list = task_list.split(";")
        path_list = (
            [None] * len(task_list) if data_path is None else data_path.split(";")
        )

    for lora_template in lora_templates:
        for task_name, data_path in zip(task_list, path_list):
            lora_config = lora_template.copy()
            if multi_task:
                lora_config["name"] = f"multi_task_{index}"
                lora_config["task_name"] = task_name
            elif task_name not in mlora.tasks.task_dict:
                assert os.path.exists(task_name), f"File '{task_name}' not exist."
                lora_config["name"] = f"casual_{index}"
                lora_config["task_name"] = "casual"
                lora_config["data"] = task_name
                lora_config["prompt"] = "alpaca"
            else:
                lora_config["name"] = (
                    f"{task_name.split(':')[-1].replace('-', '_')}_{index}"
                )
                lora_config["task_name"] = task_name

            if adapter_name is not None:
                lora_config["name"] = f"{adapter_name}_{index}"

            update_record(lora_config, "data", data_path)
            update_record(lora_config, "prompt", prompt_template)
            update_record(lora_config, "scheduler_type", lr_scheduler)
            update_record(lora_config, "warmup_steps", warmup_steps)
            update_record(lora_config, "lr", learning_rate)
            update_record(lora_config, "batch_size", batch_size)
            update_record(lora_config, "micro_batch_size", micro_batch_size)
            update_record(lora_config, "evaluate_steps", evaluate_steps)
            update_record(lora_config, "evaluate_batch_size", evaluate_batch_size)
            update_record(lora_config, "num_epochs", num_epochs)
            update_record(lora_config, "loraplus_lr_ratio", loraplus_lr_ratio)
            update_record(lora_config, "use_dora", use_dora)
            update_record(lora_config, "use_rslora", use_rslora)
            update_record(lora_config, "group_by_length", group_by_length)
            template_obj["lora"].append(lora_config)
            index += 1

    with open(config_dir, "w") as f:
        json.dump(template_obj, f, indent=4)
    print(f"Configuration file saved to {config_dir}")


def avail_tasks():
    import mlora

    print("Available task names:")
    for name in mlora.tasks.task_dict.keys():
        print(f"    {name}")
    print("These tasks can be trained and evaluated automatically using m-LoRA.")


def show_help():
    print(
        """
    Launcher of m-LoRA
    Usage: python launch.py COMMAND [ARGS...]
    Command:
        gen         generate a configuration from template
        run         start a task with existed configuration
        avail       List all available tasks
        help        Show help information

    Arguments of gen:
        --template          lora, mixlora, etc.
        --tasks             task names separate by ';'
        --adapter_name      default is task name
        --file_name         default is 'mlora.json'
        --data_path         path to input data
        --multi_task        multi-task training
        --append            append to existed config
        --prompt_template   [alpaca]
        --cutoff_len
        --save_step
        --warmup_steps
        --learning_rate
        --loraplus_lr_ratio
        --batch_size
        --micro_batch_size
        --evaluate_batch_size
        --num_epochs
        --use_dora
        --use_rslora
        --group_by_length

    Arguments of run:
        --base_model     model name or path
        --config         [mlora.json]
        --load_adapter   [false]
        --random_seed    [42]
        --cuda_device    [0]
        --log_file       [mlora.log]
        --overwrite      [false]
        --attn_impl      [eager]
        --sliding_window [false]
        --use_cache      [true]
        --quantize       [none], 4bit, 8bit
        --dtype          [bf16], fp16, fp32
        --tf32           [false]
    """
    )


command_map = {
    "gen": gen_config,
    "run": compose_command,
    "avail": avail_tasks,
    "help": show_help,
}


def main(command: str = "help", *args, **kwargs):
    command_map[command](*args, **kwargs)


if __name__ == "__main__":
    fire.Fire(main)
