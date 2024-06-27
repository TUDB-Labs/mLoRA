from typing import Dict, List, override

from .prompter import Prompter


class InstructionDataPrompter(Prompter):
    def __init__(self, template: str):
        super().__init__(template)

    @override
    def generate_prompt(self, data_points: List[Dict[str, str]]) -> List[str]:
        ret_data = []

        for data_point in data_points:
            ret_data.append(self.template_.render(data_point=data_point))

        return ret_data
