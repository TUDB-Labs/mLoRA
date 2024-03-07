import datasets
import json


def prep_data(data, file_path):
    json_data = []
    for data_point in data:
        obj = {}
        obj["instruction"] = data_point["question_stem"]
        choices_str = ""
        choices = data_point["choices"]
        for label, text in zip(choices["label"], choices["text"]):
            choices_str += f" ({label}) {text}"
        obj["input"] = choices_str
        obj["output"] = data_point["answerKey"]
        json_data.append(obj)
    with open(file_path, "w") as f:
        json.dump(json_data, f, indent=4)


obqa_data = datasets.load_dataset("allenai/openbookqa", "main")

prep_data(obqa_data["train"], "./data_train.json")
