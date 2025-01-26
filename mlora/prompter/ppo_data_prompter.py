from typing import Dict, List, override

from .prompter import Prompter


class PPODataPrompter(Prompter):
    def __init__(self, template: str):
        super().__init__(template)

    def __generate_prompt(self, data_point: Dict[str, str], optional: str) -> str:
        data = self.template_.render(data_point=data_point, Optional=optional)
        return data

    @override
    def generate_prompt(self, data_points: List[Dict[str, str]]) -> List[str]:
        instru_data = []
        chosen_data = []
        reject_data = []

        for data_point in data_points:
            data_str = self.__generate_prompt(data_point, "instruction")
            instru_data.append(data_str)
            chosen_str = self.__generate_prompt(data_point, "chosen")
            reject_str = self.__generate_prompt(data_point, "reject")
            chosen_data.append(chosen_str)
            reject_data.append(reject_str)

        data = instru_data + chosen_data + reject_data

        return data
