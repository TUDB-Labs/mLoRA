from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch

from mlora.model.args import ModelData


class PipeMessageType(Enum):
    ACTIVATIONS = "ACTIVATIONS"
    GRADIENTS = "GRADIENTS"
    COMM = "COMM"


@dataclass()
class PipeMessage:
    src_: str
    dst_: str

    msg_type_: PipeMessageType
    msg_id_: int

    tensor_data_: torch.Tensor | None
    model_data_: ModelData | None

    comm_data_: Any
