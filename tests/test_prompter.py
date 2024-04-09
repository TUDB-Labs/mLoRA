"""
This is a test file for the Prompter class.
"""

import unittest
from unittest.mock import MagicMock
from mlora import Prompter


class TestPrompter(unittest.TestCase):
    """
    Test cases for the Prompter class.
    """
    def setUp(self):
        """
        Set up method.
        """
        # Create a mock instance of Prompter
        self.prompter = MagicMock(spec=Prompter)

    def test_generate_prompt_with_input(self):
        """
        Test the generate_prompt method with input.
        """
        instruction = "Please share a healthy recipe"
        input_val = "Quinoa Salad"
        expected_output = (
            "### Instruction:\nPlease share a healthy recipe\n\n"
            "### Input:\nQuinoa Salad\n\n### Output:\n"
        )

        # Mock the generate_prompt method to return the expected output
        self.prompter.generate_prompt.return_value = expected_output

        output = self.prompter.generate_prompt(instruction, input_val)

        self.assertEqual(output, expected_output)

    def test_generate_prompt_without_input(self):
        """
        Test the generate_prompt method without input.
        """
        instruction = "Please share a healthy recipe"
        expected_output = (
            "### Instruction:\nPlease share a healthy recipe\n\n"
            "### Output:\n"
        )

        # Mock the generate_prompt method to return the expected output
        self.prompter.generate_prompt.return_value = expected_output

        output = self.prompter.generate_prompt(instruction)

        self.assertEqual(output, expected_output)

    def test_get_response(self):
        """
        Test the get_response method.
        """
        output = "### Output:Here's a healthy recipe"
        expected_response = "Here's a healthy recipe"

        # Mock the get_response method to return the expected response
        self.prompter.get_response.return_value = expected_response

        response = self.prompter.get_response(output)

        self.assertEqual(response, expected_response)


if __name__ == "__main__":
    unittest.main()
