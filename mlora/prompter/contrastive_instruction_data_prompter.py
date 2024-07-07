from typing import Dict, List, Tuple, override

from .prompter import Prompter


class ContrastiveInstructionDataPrompter(Prompter):
    def __init__(self, template: str):
        super().__init__(template)

    def __generate_prompt(self, data_point: Dict[str, str]) -> Tuple[str, str]:
        original_data = self.template_.render(data_point=data_point, is_origin=True)
        paraphrased_data = self.template_.render(data_point=data_point, is_origin=False)
        return original_data, paraphrased_data

    @override
    def generate_prompt(self, data_points: List[Dict[str, str]]) -> List[str]:
        original_data = []
        paraphrased_data = []

        for data_point in data_points:
            original_str, paraphrased_str = self.__generate_prompt(data_point)
            original_data.append(original_str)
            paraphrased_data.append(paraphrased_str)
        return original_data + paraphrased_data
