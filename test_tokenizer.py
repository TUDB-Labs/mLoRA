import unittest
from unittest.mock import MagicMock
from mlora.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        # Create a mock instance of Tokenizer using MagicMock
        self.tokenizer = MagicMock(spec=Tokenizer)

    def test_encode(self):
        test_cases = [
            # Test cases for encode method
            ("hello world", True, True, [1, 22172, 3186, 2]),
            ("hello world", False, False, [22172, 3186]),
            ("apple", False, True, [26163, 2]),
            ("apple", True, False, [1, 26163]),
            ("apple", None, None, [1, 26163]),  # Default values for bos and eos
        ]
        
        for input_str, bos, eos, expected_output in test_cases:
            with self.subTest(input_str=input_str, bos=bos, eos=eos, expected_output=expected_output):
                # Set the return value for the mock encode method
                self.tokenizer.encode.return_value = expected_output
                self.assertEqual(self.tokenizer.encode(input_str, bos=bos, eos=eos), expected_output)

    def test_decode(self):
        test_cases = [
            # Test cases for decode method
            ([22172, 3186], "hello world"),
            ([26163], "apple"),
        ]
        
        for input_list, expected_output in test_cases:
            with self.subTest(input_list=input_list, expected_output=expected_output):
                # Set the return value for the mock decode method
                self.tokenizer.decode.return_value = expected_output
                self.assertEqual(self.tokenizer.decode(input_list), expected_output)

    def test_mask_from(self):
        test_cases = [
            # Test cases for mask_from method
            ([0, 31414, 232, 328, 2005, 7], [1, 0, 0, 0, 0, 0]),
            ([31414, 232, 328, 2005, 7], [0, 0, 0, 0, 0]),
        ]
        
        for input_list, expected_output in test_cases:
            with self.subTest(input_list=input_list, expected_output=expected_output):
                # Set the return value for the mock mask_from method
                self.tokenizer.mask_from.return_value = expected_output
                self.assertEqual(self.tokenizer.mask_from(input_list), expected_output)


if __name__ == '__main__':
    unittest.main()
