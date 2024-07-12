from mlora.model.modules import LoRAFunction
from mlora.model.args import ModelData, ModelDataConfig

import torch
import unittest
import torch.nn.functional as F


class TestLoraFunction(unittest.TestCase):
    lora_a = torch.randn(8, 32, dtype=torch.float)
    lora_b = torch.randn(32, 8, dtype=torch.float)
    data = torch.randn(2, 4, 32, dtype=torch.float)
    weight = torch.randn(2, 4, 32, dtype=torch.float)

    def set_test_tensor(self):
        lora_a = self.lora_a.clone().detach().requires_grad_(True)
        lora_b = self.lora_b.clone().detach().requires_grad_(True)
        data = self.data.clone().detach().requires_grad_(True)
        weight = self.weight.clone().detach().requires_grad_(False)

        return lora_a, lora_b, data, weight

    def lora_pytorch(self):
        lora_a, lora_b, in_data, weight = self.set_test_tensor()

        data = F.dropout(in_data, 1e-4)
        data = data @ lora_a.transpose(0, 1)
        data = data @ lora_b.transpose(0, 1)
        data = data * 2.0
        data = weight + data

        loss = data.sum()
        self.py_loss = loss.item()
        loss.backward()
        self.py_grad_a = lora_a.grad
        self.py_grad_b = lora_b.grad
        self.py_grad_input = in_data.grad

    def lora_mlora(self):
        lora_a, lora_b, in_data, weight = self.set_test_tensor()
        input_args = ModelData(
            batch_tokens_=[],
            batch_mask_=[],
            data_config_=[ModelDataConfig("", "", 0, 2)],
            enable_checkpoint_=False,
            random_id_=0,
            task_name_=[""],
        )

        weight = LoRAFunction.apply(
            weight, in_data, input_args, [1e-4], [2.0], lora_a, lora_b
        )

        loss = weight.sum()
        self.mlora_loss = loss.item()
        loss.backward()
        self.mlora_grad_a = lora_a.grad
        self.mlora_grad_b = lora_b.grad
        self.mlora_grad_input = in_data.grad

    def test_lora(self):
        self.lora_pytorch()
        self.lora_mlora()

        assert abs(self.mlora_loss - self.py_loss) < 1e-3
        assert torch.allclose(self.py_grad_input, self.mlora_grad_input, 1e-4)
        assert torch.allclose(self.py_grad_b, self.mlora_grad_b, 1e-4)
        assert torch.allclose(self.py_grad_a, self.mlora_grad_a, 1e-4)


if __name__ == "__main__":
    unittest.main()
