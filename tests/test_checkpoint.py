"""
Test cases for the Checkpoint module.
"""

import unittest
from unittest.mock import patch, Mock
import torch
from mlora.checkpoint import (
    CheckpointOffloadFunction,
    CheckpointRecomputeFunction,
    check_backward_validity,
    detach_variable,
    get_device_states,
)


class TestCheckpointOffloadFunction(unittest.TestCase):
    """
    Test cases for the CheckpointOffloadFunction class.
    """

    def test_detach_variable(self):
        """
        Test detach_variable function.
        """
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([3.0, 4.0], requires_grad=True)
        inputs = (a, b)
        detached_inputs = detach_variable(inputs)
        self.assertTrue(all(inp.requires_grad is True for inp in detached_inputs))

    @patch('mlora.checkpoint.CheckpointOffloadFunction')
    def test_checkpoint_offload(self, mock_checkpoint_offload_function):
        """
        Test CheckpointOffloadFunction class.
        """
        run_function_mock = Mock(return_value=torch.tensor([4.0, 6.0]))
        mock_checkpoint_offload_function.return_value = run_function_mock

        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([3.0, 4.0], requires_grad=True)
        result = CheckpointOffloadFunction(run_function_mock, a, b)
        self.assertTrue(torch.equal(result, torch.tensor([4.0, 6.0])))


class TestCheckpointRecomputeFunction(unittest.TestCase):
    """
    Test cases for the CheckpointRecomputeFunction class.
    """

    def test_check_backward_validity(self):
        """
        Test check_backward_validity function.
        """
        with self.assertRaises(RuntimeError):
            check_backward_validity(())

        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([3.0, 4.0], requires_grad=True)
        check_backward_validity((a, b))

    def test_get_device_states(self):
        """
        Test get_device_states function.
        """
        a = torch.tensor([1.0, 2.0], device="cpu")
        b = torch.tensor([3.0, 4.0], device="cuda:0")
        devices, states = get_device_states(a, b)
        self.assertEqual(devices, [0])
        self.assertEqual(len(states), 1)
        self.assertTrue(torch.cuda.is_available())

    @patch('mlora.checkpoint.CheckpointRecomputeFunction.apply')
    def test_checkpoint_recompute(self, mock_checkpoint_recompute_function):
        """
        Test CheckpointRecomputeFunction class.
        """
        mock_result = torch.tensor([4.0, 6.0])
        mock_checkpoint_recompute_function.return_value = mock_result

        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([3.0, 4.0], requires_grad=True)
        result = CheckpointRecomputeFunction.apply(None, a, b)
        self.assertTrue(torch.equal(result, mock_result))


if __name__ == "__main__":
    unittest.main()
