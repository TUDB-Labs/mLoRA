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
    shuffle_data_cache_path_: str | None

    def __init__(self, config: TaskConfig, llm_name: str) -> None:
        self.config_ = config

        self.data_ = []
        self.now_data_idx_ = 0
        self.now_step_ = 1

        self.terminate_ = False

        self.llm_name_ = llm_name

        self.recover_folder_ = None
        self.shuffle_data_cache_path_ = None

    @abstractmethod
    def prepare(self, linears_info: OrderedDict[str, LinearInfo], tokenizer: Tokenizer):
        # task prepare for execute
        ...

    @abstractmethod
    def done(self, is_pipeline: Optional[int] = None): ...

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

    def _del_cache_file(self):
        if self.shuffle_data_cache_path_ is None:
            return
        cache_path: str = self.shuffle_data_cache_path_
        # If exist cache files, then delete them.
        if os.path.exists(cache_path):
            os.remove(cache_path)
        # If the cache folder is empty, delete it.
        dir, _ = os.path.split(cache_path)
        if os.path.exists(dir) and len(os.listdir(dir)) == 0:
            os.rmdir(dir)

    def _shuffle_data(self, data):
        # If data preprocess_type is shuffle, create a cache folder,
        # to store shuffled data and use it for saving checkpoints.
        data_name: str = ""
        if self.config_.dataset_ is not None:
            data_name = self.config_.dataset_.name_ + "_"
        # warning: The cache path can only use up to first level dirs,
        # otherwise will result in an error.
        self.shuffle_data_cache_path_ = ".cache/shuffle_" + data_name + self.task_name()
        # Clear the cache files before use.
        self._del_cache_file()
        cache_dir, _ = os.path.split(self.shuffle_data_cache_path_)
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # If exist checkpoint, copy the shuffle_data in checkpoint to cache path.
        if self.recover_folder_ is not None:
            recover_data_path: str = (
                self.context_.path_
                + os.sep
                + self.recover_folder_
                + os.sep
                + "shuffle_data"
            )
            shutil.copy(recover_data_path, self.shuffle_data_cache_path_)
            logging.info(
                "Found shuffled data successfully, data status has been restored."
            )
        return data.shuffle(
            indices_cache_file_names={"data_points": self.shuffle_data_cache_path_}
        )

    def _pre_dataset(self):
        preprocess_func: Dict[str, Callable] = {
            "default": lambda data: data,
            "shuffle": lambda data: self._shuffle_data(data),
            "sort": lambda data: data.sort(),
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

        # Process data according to the data preprocess_type.
        data = preprocess_func[preprocess_type](data)
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

    def _save_data(self, output_dir: str):
        if self.config_.dataset_ is None or self.shuffle_data_cache_path_ is None:
            return
        preprocess_type: str = self.config_.dataset_.preprocess_
        cache_path: str = self.shuffle_data_cache_path_
        # If data preprocess_type is shuffle,
        # save shuffle data from cache path to the checkpoint.
        if preprocess_type == "shuffle":
            sheffle_data_path = output_dir + os.sep + "shuffle_data"
            shutil.copy(cache_path, sheffle_data_path)

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
