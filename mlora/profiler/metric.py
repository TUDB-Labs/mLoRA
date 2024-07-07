from typing import Dict

from torch.utils.tensorboard import SummaryWriter

__metric_writer: SummaryWriter | None = None


def metric_init(path: str):
    global __metric_writer
    __metric_writer = SummaryWriter(path)


def metric_log(tag: str, value: float, step: int):
    global __metric_writer
    if __metric_writer is None:
        return

    __metric_writer.add_scalar(tag, value, step)


def metric_log_dict(tag: str, values: Dict[str, float], step: int):
    global __metric_writer
    if __metric_writer is None:
        return

    __metric_writer.add_scalars(tag, values, step)
