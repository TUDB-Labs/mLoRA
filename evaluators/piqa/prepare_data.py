import datasets
import json
import os

choices_map = ["A", "B"]


def prep_data(data, file_path):
    json_data = []
    for data_point in data:
        obj = {}
        obj["instruction"] = data_point["goal"]
        sol1 = data_point["sol1"]
        sol2 = data_point["sol2"]
        obj["input"] = f" (A) {sol1} (B) {sol2}"
        obj["output"] = choices_map[data_point["label"]]
        json_data.append(obj)
    with open(file_path, "w") as f:
        json.dump(json_data, f, indent=4)


piqa_data = datasets.load_dataset("piqa")

file_path = os.path.dirname(os.path.abspath(__file__))

prep_data(piqa_data["train"], f"{file_path}{os.sep}data_train.json")
