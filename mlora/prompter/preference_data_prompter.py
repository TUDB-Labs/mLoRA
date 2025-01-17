from typing import Dict, List, Tuple, override

from .prompter import Prompter


class PreferenceDataPrompter(Prompter):
    def __init__(self, template: str):
        super().__init__(template)

    def __generate_prompt(self, data_point: Dict[str, str]) -> Tuple[str, str]:
        chosen_data = self.template_.render(data_point=data_point, is_chosen=True)
        reject_data = self.template_.render(data_point=data_point, is_chosen=False)
        return chosen_data, reject_data

    @override
    def generate_prompt(self, data_points: List[Dict[str, str]]) -> List[str]:
        chosen_data = []
        reject_data = []

        for data_point in data_points:
            chosen_str, reject_str = self.__generate_prompt(data_point)
            chosen_data.append(chosen_str)
            reject_data.append(reject_str)

        return chosen_data + reject_data
