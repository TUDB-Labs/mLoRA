import logging
import os
import shutil
from abc import abstractmethod
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

from mlora.config import TaskConfig
from mlora.executor.context import TRAINCONTEXT_CLASS, TaskContext
from mlora.model.args import LinearInfo, Masks, MLoRADataConfig, Tokens
from mlora.model.modules import AdapterModel
from mlora.model.tokenizer import Tokenizer
from mlora.prompter import Prompter, PrompterFactory


class Task:
    config_: TaskConfig

    prompter_: Prompter
    tokenizer_: Tokenizer

    context_: TaskContext

    data_: List[Dict[str, str]]
    now_data_idx_: int
    now_step_: int

    terminate_: bool
    # need_terminal_ the llm name just for export the config file
    llm_name_: str

    recover_folder_: str | None

    def __init__(self, config: TaskConfig, llm_name: str) -> None:
        self.config_ = config

        self.data_ = []
        self.now_data_idx_ = 0
        self.now_step_ = 1

        self.terminate_ = False

        self.llm_name_ = llm_name

        self.recover_folder_ = None

    @abstractmethod
    def prepare(self, linears_info: OrderedDict[str, LinearInfo], tokenizer: Tokenizer):
        # task prepare for execute
        ...

    @abstractmethod
    def done(self): ...

    @abstractmethod
    def terminate(self): ...

    @abstractmethod
    def step(self): ...

    @abstractmethod
    def is_done(self) -> bool: ...

    @abstractmethod
    def data(self, start_idx: int) -> Tuple[List[Tokens], List[MLoRADataConfig]]: ...

    @abstractmethod
    def task_progress(self) -> int: ...

    def notify_terminate(self):
        self.terminate_ = True

    def is_terminate(self) -> bool:
        return self.terminate_

    def _pre_dataset(self):
        preprocess_func: Dict[str, Callable] = {
            "default": lambda data, _: data,
            "shuffle": lambda data, path: data.shuffle(
                indices_cache_file_names={k: path for k in data}
            ),
            "sort": lambda data, _: data.sort(),
        }

        if self.config_.dataset_ is None:
            logging.info(
                "Task dataset is empty, maybe in pipeline we do not load dataset."
            )
            return

        self.prompter_ = PrompterFactory.create(self.config_.dataset_)

        logging.info(f"Task load data from {self.config_.dataset_.data_path_}")
        data = load_dataset(
            "json", data_files={"data_points": self.config_.dataset_.data_path_}
        )

        preprocess_type = self.config_.dataset_.preprocess_
        if preprocess_type not in preprocess_func:
            raise NotImplementedError

        if preprocess_type == "shuffle":
            data_cache_path = ".cache/shuffle_data_" + self.task_name()
            if not os.path.exists(".cache"):
                os.makedirs(".cache")
            if self.recover_folder_ is not None:
                recover_data_path = (
                    self.context_.path_
                    + os.sep
                    + self.recover_folder_
                    + os.sep
                    + "shuffle_data"
                )
                shutil.copy(recover_data_path, data_cache_path)
        data = preprocess_func[preprocess_type](data, data_cache_path)
        logging.info(
            f"Adapter {self.config_.adapter_.name_} "
            f"data size: {len(data["data_points"])}"
        )

        for _, data_point in tqdm(enumerate(data["data_points"])):
            self.data_.append(data_point)

    def _pre_context(self, linears_info: OrderedDict[str, LinearInfo]):
        adapter_type = self.config_.adapter_.type_
        assert adapter_type in TRAINCONTEXT_CLASS
        self.context_ = TRAINCONTEXT_CLASS[adapter_type](
            self.config_.adapter_, linears_info
        )

    def _expand_batch_tokens(
        self, batch_tokens: List[Tokens], align_len: Optional[int] = None
    ) -> Tuple[List[Tokens], List[Masks]]:
        if align_len is None:
            align_len = max(map(lambda x: len(x), batch_tokens))

        ret_batch_tokens = []
        ret_batch_masks = []
        for tokens in batch_tokens:
            tokens, masks = self.tokenizer_.expand_tokens(tokens, align_len)
            ret_batch_tokens.append(tokens)
            ret_batch_masks.append(masks)

        return ret_batch_tokens, ret_batch_masks

    def adapter_model(self) -> List[AdapterModel]:
        return [self.context_.adapter_model()]

    def adapter_name(self) -> List[str]:
        return [self.config_.adapter_.name_]

    def task_type(self) -> str:
        return self.config_.type_

    def task_name(self) -> str:
        return self.config_.name_

    def switch_device(self, device: str):
        self.context_.switch_device(device)
