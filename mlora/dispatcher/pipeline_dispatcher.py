from mlora.dispatcher.dispatcher import Dispatcher, TrainData
from mlora.tokenizer.tokenizer import Tokenizer
from mlora.config import MLoRAConfig

import time

from typing import Dict, List


class PipelineDispatcher(Dispatcher):
    _adapter_lock_: Dict[str, bool] = {}

    def __init__(self,
                 config: MLoRAConfig,
                 tokenizer: Tokenizer) -> None:
        super().__init__(config, tokenizer)

    def activate_adapter(self, adapter_name: str):
        self._adapter_lock_[adapter_name] = True

    def deactivate_adapter(self, adapter_name: str):
        self._adapter_lock_[adapter_name] = False

    def __check_adapter_available(self, adapter_name: str) -> bool:
        if adapter_name in self._adapter_lock_:
            return self._adapter_lock_[adapter_name]
        return True

    def rigister_strategies(self):
        self.rigister_strategy("pipe", self.pipe_dispatch_strategy)

    def pipe_dispatch_strategy(self) -> Dict[str, List[TrainData]]:
        ret_train_data = {}
        cnt = 0
        for task in self.running_train_task_:
            assert not task.is_train_done()

            # check the adapter is available
            if not self.__check_adapter_available(task.adapter_name_):
                continue
            self.deactivate_adapter(task.adapter_name_)
            ret_train_data[task.adapter_name_] = task.get_train_data()
            cnt += 1
            if cnt >= self.train_lora_simultaneously_num_:
                break
        time.sleep(1 / 100000)
        return ret_train_data
