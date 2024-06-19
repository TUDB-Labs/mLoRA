import yaml
from typing import Dict, List

from .dispatcher import DispatcherConfig
from .dataset import DatasetConfig
from .adapter import AdapterConfig, ADAPTERCONFIG_CLASS
from .task import TaskConfig, TASKCONFIG_CLASS


class MLoRAConfig:
    dispatcher_: DispatcherConfig = None
    tasks_: List[TaskConfig] = []
    __datasets_: Dict[str, DatasetConfig] = {}
    __adapters_: Dict[str, AdapterConfig] = {}

    def __init_datasets(self, config: List[Dict[str, any]]):
        for item in config:
            name = item["name"]
            self.__datasets_[name] = DatasetConfig(item)

    def __init_adapters(self, config: List[Dict[str, any]]):
        for item in config:
            name = item["name"]
            atype = item["type"]
            self.__adapters_[name] = ADAPTERCONFIG_CLASS[atype](item)

    def __init_tasks(self, config: List[Dict[str, any]]):
        for item in config:
            assert item["type"] in TASKCONFIG_CLASS
            self.tasks_.append(TASKCONFIG_CLASS[item["type"]](
                item, self.__adapters_, self.__datasets_))

    def __init__(self, path: str):
        with open(path) as fp:
            config = yaml.safe_load(fp)

        self.dispatcher_ = DispatcherConfig(config["dispatcher"])

        # must ensure the adapter and datasets init before the task
        self.__init_datasets(config["datasets"])
        self.__init_adapters(config["adapters"])

        self.__init_tasks(config["tasks"])


class MLoRAServerConfig(MLoRAConfig):

    def __init__(self, config: Dict[str, str]) -> None:
        self.dispatcher_ = DispatcherConfig(config)
