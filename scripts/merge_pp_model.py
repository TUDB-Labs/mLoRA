import torch
import os
import re
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='m-LoRA merge pipeline model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='local path of the saved pipeline model')
    args = parser.parse_args()
    model_path = args.model_path

    pattern = re.compile(r'^adapter_model_.*\.bin$')
    files = os.listdir(model_path)
    matched_files = [file for file in files if pattern.match(file)]

    total_model = {}
    for file in matched_files:
        path = os.path.join(model_path, file)
        model = torch.load(path, map_location='cpu')
        total_model.update(model)

    torch.save(total_model, os.path.join(model_path, 'adapter_model.bin'))
