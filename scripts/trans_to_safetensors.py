import torch
import json
import os
import argparse
from safetensors.torch import save_file


def check_safetensor_support(path: str):
    # try find model.safetensors.index.json
    safetensors_json_path = os.path.join(path, "model.safetensors.index.json")
    return os.path.exists(safetensors_json_path)


def trans_to_safetensor(path: str):
    pt_json_path = os.path.join(path, "pytorch_model.bin.index.json")
    safetensors_json_path = os.path.join(path, "model.safetensors.index.json")
    weights_paths = set([])

    with open(pt_json_path, "r") as f_origin:
        json_data = json.load(f_origin)

    # trans weigth_map
    for weight_name, weight_path in json_data["weight_map"].items():
        postfix = weight_path.split(".")[-1]
        json_data["weight_map"][weight_name] = weight_path.replace(f".{postfix}", ".safetensors")
        weights_paths.add(os.path.join(path, weight_path))

    # save json data to safetensors_json_path
    print(f"save safetensors index to {safetensors_json_path}...")
    with open(safetensors_json_path, "w") as f:
        f.write(json.dumps(json_data, indent=2))
    print(f"save safetensors index to {safetensors_json_path} done")

    # for loading transformer model
    metadata = {"format": "pt"}

    # trans pt to safetensors
    for weight_path in weights_paths:
        postfix = weight_path.split(".")[-1]
        weight_safetensors_path = weight_path.replace(f".{postfix}", ".safetensors")
        print(f"trans {weight_path} to {weight_safetensors_path}...")
        weight = torch.load(weight_path)
        save_file(weight, weight_safetensors_path, metadata)
        print(f"trans {weight_path} to {weight_safetensors_path} done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='m-LoRA trans to safetensors program')
    parser.add_argument('--model_path', type=str, required=True,
                        help='local path of the base model')
    args = parser.parse_args()
    model_path = args.model_path

    if check_safetensor_support(model_path):
        print("this model already support safetensors")
        exit(0)

    trans_to_safetensor(model_path)
