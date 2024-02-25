from mlora.model.modelargs import MultiLoraBatchData

import torch


from dataclasses import dataclass
from enum import Enum


class PipeMessageType(Enum):
    ACTIVATIONS = "ACTIVATIONS"
    GRADIENTS = "GRADIENTS"


@dataclass()
class PipeMessage:
    src_: str
    dst_: str

    msg_type_: PipeMessageType
    msg_id_: int

    tensor_data_: torch.Tensor
    batch_data_: MultiLoraBatchData
