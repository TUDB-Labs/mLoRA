import torch

from mlora.model.args import LLMModelArgs


class OutputLayer(torch.nn.Module):
    def __init__(self, weight: torch.Tensor, args: LLMModelArgs):
        super().__init__()
        self.lm_head_ = torch.nn.Linear(
            args.dim_,
            args.vocab_size_,
            bias=False,
            device=args.device_,
            dtype=args.dtype_,
        )

        with torch.no_grad():
            if weight.device == torch.device("meta"):
                self.lm_head_.weight = weight
            else:
                self.lm_head_.weight.copy_(weight)
        self.lm_head_.requires_grad_(False)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.lm_head_(data).float()
