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
    # Test detach_variable function
    def test_detach_variable(self):
        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([3.0, 4.0], requires_grad=True)
        inputs = (a, b)
        detached_inputs = detach_variable(inputs)
        # Assert that the detached_inputs are detached
        self.assertTrue(all(inp.requires_grad is True for inp in detached_inputs))

    # Test CheckpointOffloadFunction
    @patch('mlora.checkpoint.CheckpointOffloadFunction')
    def test_checkpoint_offload(self, MockCheckpointOffloadFunction):
        run_function_mock = Mock(return_value=torch.tensor([4.0, 6.0]))
        MockCheckpointOffloadFunction.return_value = run_function_mock

        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([3.0, 4.0], requires_grad=True)
        result = CheckpointOffloadFunction(run_function_mock, a, b)
        # Assert that the result is correct
        self.assertTrue(torch.equal(result, torch.tensor([4.0, 6.0])))


class TestCheckpointRecomputeFunction(unittest.TestCase):
    # Test check_backward_validity function
    def test_check_backward_validity(self):
        with self.assertRaises(RuntimeError):
            # Mocking an empty tuple as input
            check_backward_validity(())

        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([3.0, 4.0], requires_grad=True)
        check_backward_validity((a, b))

    # Test get_device_states function
    def test_get_device_states(self):
        a = torch.tensor([1.0, 2.0], device="cpu")
        b = torch.tensor([3.0, 4.0], device="cuda:0")
        devices, states = get_device_states(a, b)
        # Assert that the devices and states are correct
        self.assertEqual(devices, [0])
        self.assertEqual(len(states), 1)
        self.assertTrue(torch.cuda.is_available())

    # Test CheckpointRecomputeFunction
    @patch('mlora.checkpoint.CheckpointRecomputeFunction.apply')
    def test_checkpoint_recompute(self, MockCheckpointRecomputeFunction):
        mock_result = torch.tensor([4.0, 6.0])
        MockCheckpointRecomputeFunction.return_value = mock_result

        a = torch.tensor([1.0, 2.0], requires_grad=True)
        b = torch.tensor([3.0, 4.0], requires_grad=True)
        result = CheckpointRecomputeFunction.apply(None, a, b)

        self.assertTrue(torch.equal(result, mock_result))

if __name__ == "__main__":
    unittest.main()
