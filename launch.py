#!/usr/bin/env python3
import tempfile
import json
import fire
import os

file_path = ".launcher"
work_path = os.path.dirname(os.path.abspath(__file__))


def compose_command(base_model: str,
                    config: str = "mlora.json",
                    model_dtype: str = "16bit",
                    load_adapter: bool = False,
                    random_seed: int = 42,
                    log_file: str = "mlora.log",
                    overwrite: bool = False,
                    cuda_device: int = 0):
    assert model_dtype not in ["4bit", "8bit" "16bit"]
    command = f"CUDA_VISIBLE_DEVICES={cuda_device} python mlora.py"
    command += f" --base_model {base_model}"
    command += f" --config {config}"
    command += f" --load_{model_dtype}"
    if load_adapter:
        command += " --load_adapter"
    command += f" --seed {random_seed}"
    command += f" --log_file {log_file}"
    if overwrite:
        command += " --overwrite"
    return command


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

    os.system(compose_command(config=config, **kwargs))
    if len(evaluate_config["lora"]) > 0:
        temp = tempfile.NamedTemporaryFile("w+t")
        json.dump(evaluate_config, temp, indent=4)
        temp.flush()
        os.system(compose_command(config=temp.name, **kwargs) + " --evaluate")


def gen_config(template_name: str,
               task_names: str,
               file_name: str = "mlora.json",
               cutoff_len: int = 512,
               save_step: int = 1000,
               warmup_steps: float = 0.2,
               learning_rate: float = 1e-4,
               batch_size: int = 16,
               micro_batch_size: int = 4,
               test_batch_size: int = 16,
               num_epochs: int = 2,
               group_by_length: bool = False):
    import mlora
    template_name = f"{work_path}{os.sep}{file_path}{os.sep}{template_name}.json"
    with open(template_name, 'r', encoding='utf8') as fp:
        template_obj = json.load(fp)
    template_obj["cutoff_len"] = cutoff_len
    template_obj["save_step"] = save_step
    lora_templates = template_obj["lora"]
    template_obj["lora"] = []
    index = 0
    for lora_template in lora_templates:
        for task_name in task_names.split(';'):
            lora_config = lora_template.copy()
            casual_task = (task_name not in mlora.task_dict)
            if casual_task:
                lora_config["name"] = f"casual_{index}"
                lora_config["task_name"] = "casual"
                lora_config["data"] = task_name
                lora_config["prompt"] = "template/alpaca.json"
            else:
                lora_config["name"] = f"{task_name.split(':')[-1].replace('-', '_')}_{index}"
                lora_config["task_name"] = task_name
            lora_config["warmup_steps"] = warmup_steps
            lora_config["lr"] = learning_rate
            lora_config["batch_size"] = batch_size
            lora_config["micro_batch_size"] = micro_batch_size
            lora_config["test_batch_size"] = test_batch_size
            lora_config["num_epochs"] = num_epochs
            lora_config["group_by_length"] = group_by_length
            template_obj["lora"].append(lora_config)
            index += 1

    config_dir = f"{work_path}{os.sep}{file_name}"
    with open(config_dir, "w") as f:
        json.dump(template_obj, f, indent=4)
    print(f"Configuration file saved to {config_dir}")


def avail_tasks():
    import mlora
    print("Available task names:")
    for name in mlora.task_dict.keys():
        print(f"    {name}")
    print("These tasks can be trained and evaluated automatically using m-LoRA.")


def show_help():
    print("Launcher of m-LoRA")
    print("Usage: python launch.py COMMAND [ARGS...]")
    print("Command:")
    print("    gen         generate a configuration from template")
    print("    run         Automatically training and evaluate")
    print("    evaluate    Run evaluation on existed adapter")
    print("    train       Run training with configuration")
    print("    avail       List all available tasks")
    print("    help        Show help information")
    print("")
    print("Arguments of gen:")
    print("    --template_name    lora, mixlora, mixlora_compare")
    print("    --task_names       task names separate by \';\'")
    print("    --file_name        [mlora.json]")
    print("    --cutoff_len       [512]")
    print("    --save_step        [1000]")
    print("    --warmup_steps     [0.2]")
    print("    --learning_rate    [1e-4]")
    print("    --batch_size       [16]")
    print("    --micro_batch_size [4]")
    print("    --test_batch_size  [16]")
    print("    --num_epochs       [2]")
    print("    --group_by_length  [false]")
    print("")
    print("Arguments of run, train and evaluate:")
    print("    --base_model   model name or path")
    print("    --config       [mlora.json]")
    print("    --model_dtype  [16bit], 8bit, 4bit")
    print("    --load_adapter [false]")
    print("    --random_seed  [42]")
    print("    --log_file     [mlora.log]")
    print("    --overwrite    [false]")
    print("    --cuda_device  [0]")
    print("")


command_map = {
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
