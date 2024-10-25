from typing import Dict


class DictConfig:
    __params_map: Dict[str, str] = {}

    def __init__(self, config: Dict[str, str]) -> None:
        self.init(self.__params_map, config)

    def init(self, params_map: Dict[str, str], config: Dict[str, str]):
        for key, value in params_map.items():
            if key in config:
                setattr(self, key, config[key])
            else:
                setattr(self, key, None)
