import torch


class RMSNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.norm_eps_ = eps
        self.weight_ = weight

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        input_dtype = data.dtype

        v = data.to(torch.float32).pow(2).mean(-1, keepdim=True)
        rv = data * torch.rsqrt(v + self.norm_eps_)

        return (self.weight_ * rv).to(input_dtype)
