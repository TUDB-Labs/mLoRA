import unittest
from unittest.mock import patch, Mock
import torch
from mlora.generate import GenerateConfig, gen_outputs, logits_process, generate

class TestGenerate(unittest.TestCase):
    def setUp(self):
        self.model = Mock()
        # Mocking the Tokenizer class
        with patch("mlora.Tokenizer") as MockTokenizer:
            self.tokenizer_instance = MockTokenizer.return_value
            self.tokenizer_instance.encode.return_value = [1, 2, 3]  # Example return value
            self.tokenizer_instance.decode.return_value = "decoded_text"  # Example return value

        self.configs = [
            GenerateConfig(adapter_name="test_adapter", prompts=["Test prompt."])
        ]
        self.max_gen_len = 128

    def test_logits_process(self):
        probs = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]])
        prev_tokens = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])
        result = logits_process(probs, prev_tokens)
        expected_shape = (2,)
        self.assertEqual(result.shape, expected_shape)

    def test_gen_outputs(self):
        prompts = ["Test prompt."]
        tokens = torch.tensor([[1, 2]])
        result = gen_outputs(self.configs, self.tokenizer_instance, prompts, tokens, self.max_gen_len)
        expected_result = {'test_adapter': []}
        self.assertEqual(result, expected_result)

if __name__ == "__main__":
    unittest.main()
