import torch


def is_offload_device(device: torch.device):
    return device == torch.device("meta")
