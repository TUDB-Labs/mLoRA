from collections import OrderedDict

from mlora.config import AdapterConfig
from mlora.model.args import LinearInfo

from .context import TaskContext


class InferenceTaskContext(TaskContext):
    def __init__(
        self, config: AdapterConfig, linears_info: OrderedDict[str, LinearInfo]
    ) -> None:
        super().__init__(config)

        # load the adapter's weight
        self.load_weight(linears_info)

        # disable all the weights' grad
        for module in self.adapter_model_.values():
            module.disable_grad()

    def switch_device(self, device: str) -> None:
        if self.device_ == device:
            return

        for _, adapter in self.adapter_model_.items():
            self.switch_list_tensor(adapter.get_all_tensors(), device)

        self.device_ = device

    def step(self) -> None: ...
