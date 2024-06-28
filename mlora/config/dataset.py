from typing import Dict

from .config import DictConfig


class DatasetConfig(DictConfig):
    name_: str
    data_path_: str
    prompt_path_: str
    prompt_type_: str
    preprocess_: str

    __params_map: Dict[str, str] = {
        "name_": "name",
        "data_path_": "data",
        "prompt_path_": "prompt",
        "prompt_type_": "prompt_type",
        "preprocess_": "preprocess",
    }

    def __init__(self, config: Dict[str, str]):
        super().__init__(config)
        self.init(self.__params_map, config)
