from aspen import DataSet

import json
import unittest

from typing import List

with open('config/lora.json', 'r', encoding='utf8') as fp:
    config = json.load(fp)


class MockTokenizer:
    def __init__(self):
        self.pad_id_ = 0

    def encode(self, text: str, bos: bool, eos: bool) -> List[int]:
        return [0] * len(text)


class TestDataSet(unittest.TestCase):
    def test_load_dataset(self):
        dataset = DataSet(config, MockTokenizer())
        input_data = dataset.get_batch_data()
        self.assertEqual(len(input_data.prompts_), 4)
        for p in input_data.prompts_:
            if "Input" in p:
                self.assertEqual(
                    "### Instruction:\nInstruction demo.\n\n### Input:\nInput demo.\n\n### Output:\nOutput demo.\n", p)
            else:
                self.assertEqual(
                    "### Instruction:\nInstruction demo.\n\n### Output:\nOutput demo.\n", p)
        self.assertEqual(len(input_data.lora_batch_data_config_), 2)
        self.assertEqual(input_data.lora_batch_data_config_[
                         0].batch_start_idx_, 0)
        self.assertEqual(input_data.lora_batch_data_config_[
                         0].batch_end_idx_, 2)
        self.assertEqual(input_data.lora_batch_data_config_[
                         1].batch_start_idx_, 2)
        self.assertEqual(input_data.lora_batch_data_config_[
                         1].batch_end_idx_, 4)


if __name__ == '__main__':
    unittest.main()
