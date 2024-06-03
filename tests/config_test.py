from mlora.config.mlora import MLoRAConfig

import unittest

from typing import List


class ConfigTest(unittest.TestCase):
    def test_load_config(self):
        mlora_config: MLoRAConfig = MLoRAConfig("demo/dummy.json")
        print(mlora_config)


if __name__ == '__main__':
    unittest.main()
