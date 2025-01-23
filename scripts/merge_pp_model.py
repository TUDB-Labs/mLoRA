import torch
import os
import re
import argparse
import shutil


def get_rank_folders(base_path):
    rank_folders = []
    for item in os.listdir(base_path):
        full_path = os.path.join(base_path, item)
        if os.path.isdir(full_path) and re.match(r"^rank_\d+$", item):
            rank_folders.append(full_path)
    return rank_folders


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="m-LoRA merge pipeline model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="local path of the saved pipeline model. e.g. python scripts/merge_pp_model.py --model_path adapters/lora_sft",
    )
    args = parser.parse_args()
    model_path = args.model_path
    rank_folders = get_rank_folders(model_path)

    matched_files = []
    for folder in rank_folders:
        adapter_model_path = os.path.join(folder, "adapter_model.bin")
        if os.path.isfile(adapter_model_path):
            matched_files.append(adapter_model_path)

    total_model = {}
    for file in matched_files:
        model = torch.load(file, map_location="cpu")
        total_model.update(model)

    torch.save(total_model, os.path.join(model_path, "adapter_model.bin"))

    # Copy adapter_config.json from one of the rank_* folders to model_path
    if rank_folders:
        config_src_path = os.path.join(rank_folders[0], "adapter_config.json")
        config_dst_path = os.path.join(model_path, "adapter_config.json")
        if os.path.isfile(config_src_path):
            shutil.copy(config_src_path, config_dst_path)
