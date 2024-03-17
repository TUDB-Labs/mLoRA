import json
import fire
import os

file_path = ".launcher"
work_path = os.path.dirname(os.path.abspath(__file__))


def evaluate(base_model: str,
             config: str = "launcher.json",
             model_dtype: str = "16bit",
             random_seed: int = 42,
             log_file: str = "launcher.log",
             cuda_device: int = 0):
    assert model_dtype not in ["4bit", "8bit" "16bit"]
    command = f"CUDA_VISIBLE_DEVICES={cuda_device} python mlora.py"
    command += f" --base_model {base_model}"
    command += f" --config {config}"
    command += f" --load_{model_dtype}"
    command += f" --seed {random_seed}"
    command += f" --log_file {log_file}"
    os.system(command + " --evaluate")


def train(base_model: str,
          config: str = "launcher.json",
          model_dtype: str = "16bit",
          load_adapter: bool = False,
          random_seed: int = 42,
          log_file: str = "launcher.log",
          overwrite: bool = False,
          cuda_device: int = 0):
    assert model_dtype not in ["4bit", "8bit" "16bit"]
    command = f"CUDA_VISIBLE_DEVICES={cuda_device} python mlora.py"
    command += f" --base_model {base_model}"
    command += f" --config {config}"
    command += f" --load_{model_dtype}"
    if load_adapter:
        command += f" --load_adapter"
    command += f" --seed {random_seed}"
    command += f" --log_file {log_file}"
    if overwrite:
        command += f" overwrite"
    os.system(command)


def run_task(base_model: str,
             config: str = "launcher.json",
             model_dtype: str = "16bit",
             load_adapter: bool = False,
             random_seed: int = 42,
             log_file: str = "launcher.log",
             overwrite: bool = False,
             cuda_device: int = 0):
    assert model_dtype not in ["4bit", "8bit" "16bit"]
    command = f"CUDA_VISIBLE_DEVICES={cuda_device} python mlora.py"
    command += f" --base_model {base_model}"
    command += f" --config {config}"
    command += f" --load_{model_dtype}"
    if load_adapter:
        command += f" --load_adapter"
    command += f" --seed {random_seed}"
    command += f" --log_file {log_file}"
    if overwrite:
        command += f" overwrite"
    os.system(command)
    os.system(command + " --evaluate")


def gen_config(template_name: str,
               task_names: str,
               file_name: str = "launcher.json",
               cutoff_len: int = 512,
               save_step: int = 1000,
               warmup_steps: float = 0.2,
               learning_rate: float = 1e-4,
               batch_size: int = 16,
               micro_batch_size: int = 4,
               test_batch_size: int = 16,
               num_epochs: int = 2,
               group_by_length: bool = False):

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
    print("m-LoRA launcher")
    print("usage: python launcher.py COMMAND [ARGS...]")
    print("command:")
    print("    evaluate    Run evaluation on existed adapter")
    print("    train       Run training with configuration")
    print("    run-task    Automatically training and evaluate")
    print("    Arguments:")
    print("        --base_model   model name or path")
    print("        --config       [launcher.json]")
    print("        --model_dtype  [16bit], 8bit, 4bit")
    print("        --load_adapter [false]")
    print("        --random_seed  [42]")
    print("        --log_file     [launcher.log]")
    print("        --overwrite    [false]")
    print("        --cuda_device  [0]")
    print("")
    print("    avail-tasks    List all available tasks")
    print("")
    print("    gen-config     generate a configuration from template")
    print("    Arguments:")
    print("        --template_name    lora, mixlora, mixlora_compare")
    print("        --task_names       task names separate by \';\'")
    print("        --file_name        [launcher.json]")
    print("        --cutoff_len       [512]")
    print("        --save_step        [1000]")
    print("        --warmup_steps     [0.2]")
    print("        --learning_rate    [1e-4]")
    print("        --batch_size       [16]")
    print("        --micro_batch_size [4]")
    print("        --test_batch_size  [16]")
    print("        --num_epochs       [2]")
    print("        --group_by_length  [false]")
    print("")
    print("    help")
    print("")


command_map = {
    "evaluate": evaluate,
    "train": train,
    "run-task": run_task,
    "avail-tasks": avail_tasks,
    "gen-config": gen_config,
    "help": show_help,
}


def main(command: str = "help", **kwargs):
    command_map[command](**kwargs)


if __name__ == "__main__":
    fire.Fire(main)
