from mlora.dispatcher.dispatcher import Dispatcher, TrainData
from mlora.tokenizer.tokenizer import Tokenizer
from mlora.config import MLoRAConfig

from typing import Dict, List


class PipelineDispatcher(Dispatcher):
    _adapter_backward_cnt_: Dict[str, int] = {}
    _adapter_forward_cnt_: Dict[str, int] = {}
    _adapter_accumulation_step_: Dict[str, int] = {}

    def __init__(self,
                 config: MLoRAConfig,
                 tokenizer: Tokenizer) -> None:
        super().__init__(config, tokenizer)
        for lora_config in config.lora_configs_:
            adapter_name = lora_config.adapter_name_
            accumulation_step = lora_config.batch_size_ / lora_config.micro_batch_size_
            self._adapter_forward_cnt_[adapter_name] = 0
            self._adapter_backward_cnt_[adapter_name] = 0
            self._adapter_accumulation_step_[adapter_name] = accumulation_step

    def update_backward_cnt(self, adapter_name: str):
        self._adapter_backward_cnt_[adapter_name] += 1
        if self._adapter_backward_cnt_[adapter_name] == self._adapter_accumulation_step_[adapter_name]:
            self._adapter_forward_cnt_[adapter_name] = 0
            self._adapter_backward_cnt_[adapter_name] = 0

    def update_forward_cnt(self, adapter_name: str):
        self._adapter_forward_cnt_[adapter_name] += 1

    def __check_adapter_available(self, adapter_name: str) -> bool:
        return self._adapter_forward_cnt_[adapter_name] < self._adapter_accumulation_step_[adapter_name]

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
            self.update_forward_cnt(task.adapter_name_)
            ret_train_data[task.adapter_name_] = task.get_train_data()
            cnt += 1
            if cnt >= self.train_lora_simultaneously_num_:
                break

        return ret_train_data
