from mlora.dispatcher.dispatcher import TrainTask
from mlora.config import MLoRAConfig

import unittest

from typing import List


class MockTokenizer:
    def __init__(self):
        self.pad_id_ = 0

    def encode(self, text: str, bos: bool, eos: bool) -> List[int]:
        return [0] * len(text)


class TestDataSet(unittest.TestCase):
    def test_load_dataset(self):
        tasks: List[TrainTask] = []
        mlora_config: MLoRAConfig = MLoRAConfig("config/finetune.json")

        for lora_config in mlora_config.lora_configs_:

            task = TrainTask(MockTokenizer(), lora_config)
            task.load_data()
            tasks.append(task)
            self.assertEqual(len(task.train_token_data_), 2)

        self.assertEqual(len(tasks), 3)


if __name__ == '__main__':
    unittest.main()
