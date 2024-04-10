#!/usr/bin/env python3
import tempfile
import json
import fire
import os

file_path = ".launcher"
work_path = os.path.dirname(os.path.abspath(__file__))


def compose_command(base_model: str,
                    config: str = "mlora.json",
                    load_adapter: bool = False,
                    random_seed: int = 42,
                    cuda_device: int = None,
                    log_file: str = "mlora.log",
                    overwrite: bool = False,
                    attn_impl: str = None,
                    quantize: str = None,
                    dtype: str = "bf16",
                    tf32: bool = False):
    assert quantize in (None, "4bit", "8bit")
    assert dtype in ("fp32", "fp16", "bf16")
    command = "python mlora.py"
    if cuda_device is not None:
        command = f"CUDA_VISIBLE_DEVICES={cuda_device} " + command
    command += f" --base_model {base_model}"
    command += f" --config {config}"
    if load_adapter:
        command += " --load_adapter"
    command += f" --seed {random_seed}"
    command += f" --log_file {log_file}"
    if overwrite:
        command += " --overwrite"
    if attn_impl is not None:
        command += f" --attn_impl {attn_impl}"
    if quantize is not None:
        command += f" --load_{quantize}"
    if dtype in ("fp16", "bf16"):
        command += f" --{dtype}"
    if tf32:
        command += " --tf32"
    return command


def inference(**kwargs):
    os.system(compose_command(**kwargs) + " --inference")


def evaluate(**kwargs):
    os.system(compose_command(**kwargs) + " --evaluate")


def train(**kwargs):
    os.system(compose_command(**kwargs))


def run(config: str = "mlora.json", **kwargs):
    config = f"{work_path}{os.sep}{config}"
    with open(config, 'r', encoding='utf8') as fp:
        config_obj = json.load(fp)
    evaluate_config = config_obj.copy()
    evaluate_config["lora"] = []
    for lora_config in config_obj["lora"]:
        if lora_config["task_name"] != "casual":
            evaluate_config["lora"].append(lora_config)

    if os.system(compose_command(config=config, **kwargs)) != 0:
        return

    if len(evaluate_config["lora"]) > 0:
        temp = tempfile.NamedTemporaryFile("w+t")
        json.dump(evaluate_config, temp, indent=4)
        temp.flush()
        os.system(compose_command(config=temp.name, **kwargs) + " --evaluate")


def update_record(dict_: dict, key_, value_):
    if value_ is not None:
        dict_[key_] = value_


def gen_config(
        # essential
        template: str,
        tasks: str,
        # optional
        adapter_name: str = None,
        file_name: str = "mlora.json",
        # default value provided by template
        cutoff_len: int = None,
        save_step: int = None,
        lr_scheduler: str = None,
        warmup_steps: float = None,
        learning_rate: float = None,
        batch_size: int = None,
        micro_batch_size: int = None,
        test_batch_size: int = None,
        num_epochs: int = None,
        loraplus_lr_ratio: float = None,
        use_dora: bool = None,
        use_rslora: bool = None,
        multi_task: bool = None,
        group_by_length: bool = None):
    import mlora
    template = f"{work_path}{os.sep}{file_path}{os.sep}{template}.json"
    with open(template, 'r', encoding='utf8') as fp:
        template_obj = json.load(fp)
    update_record(template_obj, "cutoff_len", cutoff_len)
    update_record(template_obj, "save_step", save_step)
    lora_templates = template_obj["lora"]
    template_obj["lora"] = []
    index = 0
    if multi_task:
        tasks = [tasks]
    else:
        tasks = tasks.split(';')

    for lora_template in lora_templates:
        for task_name in tasks:
            lora_config = lora_template.copy()
            casual_task = (
                not multi_task and task_name not in mlora.tasks.task_dict)
            if casual_task:
                lora_config["name"] = f"casual_{index}"
                lora_config["task_name"] = "casual"
                lora_config["data"] = task_name
                lora_config["prompt"] = "template/alpaca.json"
            else:
                lora_config["name"] = f"{task_name.split(':')[-1].replace('-', '_')}_{index}"
                lora_config["task_name"] = task_name

            if adapter_name is not None:
                lora_config["name"] = f"{adapter_name}_{index}"

            update_record(lora_config, "scheduler_type", lr_scheduler)
            update_record(lora_config, "warmup_steps", warmup_steps)
            update_record(lora_config, "lr", learning_rate)
            update_record(lora_config, "batch_size", batch_size)
            update_record(lora_config, "micro_batch_size", micro_batch_size)
            update_record(lora_config, "test_batch_size", test_batch_size)
            update_record(lora_config, "num_epochs", num_epochs)
            update_record(lora_config, "loraplus_lr_ratio", loraplus_lr_ratio)
            update_record(lora_config, "use_dora", use_dora)
            update_record(lora_config, "use_rslora", use_rslora)
            update_record(lora_config, "group_by_length", group_by_length)
            template_obj["lora"].append(lora_config)
            index += 1

    config_dir = f"{work_path}{os.sep}{file_name}"
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
    print("Launcher of m-LoRA")
    print("Usage: python launch.py COMMAND [ARGS...]")
    print("Command:")
    print("    gen         generate a configuration from template")
    print("    run         Automatically training and evaluate")
    print("    inference   Run inference on existed adapter")
    print("    evaluate    Run evaluation on existed adapter")
    print("    train       Run training with configuration")
    print("    avail       List all available tasks")
    print("    help        Show help information")
    print("")
    print("Arguments of gen:")
    print("    --template          lora, mixlora, etc.")
    print("    --tasks             task names separate by \';\'")
    print("    --adapter_name      default is task name")
    print("    --file_name         [mlora.json]")
    print("    --cutoff_len")
    print("    --save_step")
    print("    --warmup_steps")
    print("    --learning_rate")
    print("    --loraplus_lr_ratio")
    print("    --batch_size")
    print("    --micro_batch_size")
    print("    --test_batch_size")
    print("    --num_epochs")
    print("    --use_dora")
    print("    --use_rslora")
    print("    --group_by_length")
    print("")
    print("Arguments of run, train, inference and evaluate:")
    print("    --base_model   model name or path")
    print("    --config       [mlora.json]")
    print("    --load_adapter [false]")
    print("    --random_seed  [42]")
    print("    --cuda_device  [0]")
    print("    --log_file     [mlora.log]")
    print("    --overwrite    [false]")
    print("    --attn_impl    [eager]")
    print("    --quantize     [none], 4bit, 8bit")
    print("    --dtype        [bf16], fp16, fp32")
    print("    --tf32         [false]")
    print("")


command_map = {
    "inference": inference,
    "evaluate": evaluate,
    "train": train,
    "run": run,
    "avail": avail_tasks,
    "gen": gen_config,
    "help": show_help,
}


def main(command: str = "help", **kwargs):
    command_map[command](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
