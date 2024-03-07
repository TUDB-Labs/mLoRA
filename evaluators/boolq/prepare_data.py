import datasets
import json


def prep_data(data, file_path):
    json_data = []
    for data_point in data:
        obj = {}
        obj["instruction"] = data_point["question"]
        obj["input"] = data_point["passage"]
        obj["output"] = "yes" if data_point["answer"] else "no"
        json_data.append(obj)
    with open(file_path, "w") as f:
        json.dump(json_data, f, indent=4)


boolq_data = datasets.load_dataset("google/boolq")

prep_data(boolq_data["train"], "./data_train.json")
