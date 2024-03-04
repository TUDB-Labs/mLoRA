from mlora.dispatcher.dispatcher import TrainTask

import json
import unittest

from typing import List

with open('config/finetune.json', 'r', encoding='utf8') as fp:
    config = json.load(fp)


class MockTokenizer:
    def __init__(self):
        self.pad_id_ = 0

    def encode(self, text: str, bos: bool, eos: bool) -> List[int]:
        return [0] * len(text)


class TestDataSet(unittest.TestCase):
    def test_load_dataset(self):
        tasks: List[TrainTask] = []

        for lora in config["lora"]:
            task = TrainTask(MockTokenizer(),
                             lora["name"],
                             lora["data"],
                             -1,
                             lora["test_data"],
                             lora["prompt"],
                             lora["num_epochs"],
                             lora["batch_size"],
                             lora["micro_batch_size"],
                             lora["test_batch_size"],
                             config["cutoff_len"],
                             config.get("group_by_length", False))
            task.load_data()
            tasks.append(task)
            self.assertEqual(len(task.train_token_data_), 2)

        self.assertEqual(len(tasks), 3)


if __name__ == '__main__':
    unittest.main()
