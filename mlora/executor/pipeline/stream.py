import time

import torch


class CudaStream:
    stream_: torch.cuda.Stream
    event_: torch.cuda.Event

    def __init__(self, stream: torch.cuda.Stream) -> None:
        self.stream_ = stream
        self.event_ = torch.cuda.Event()

    def poll(self) -> None:
        self.event_.record(stream=self.stream_)
        while not self.event_.query():
            time.sleep(1 / 1000)
