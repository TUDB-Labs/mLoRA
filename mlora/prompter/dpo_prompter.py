from .prompter import Prompter


from typing import Dict, List, Tuple


class DPOPrompter(Prompter):
    def __init__(self, template: str):
        super().__init__(template)

    def __generate_prompt_by_output(self, data: Dict[str, str]) -> str:
        try:
            ret_val = self.template_["prompt"].format(**data)
        except:
            ret_val = ""

        if ret_val != "":
            return ret_val

        return self.template_["prompt_no_input"].format(**data)

    def generate_prompt(self, data: Dict[str, str]) -> Tuple[str, str]:
        chosen_data = {**data, "output": data["chosen"]}
        reject_data = {**data, "output": data["reject"]}

        return self.__generate_prompt_by_output(chosen_data), self.__generate_prompt_by_output(reject_data)

    def generate_prompt_batch(self, datas: List[Dict[str, str]]) -> List[str]:
        chosen_data = []
        reject_data = []
        for data in datas:
            chosen_str, reject_str = self.generate_prompt(data)
            chosen_data.append(chosen_str)
            reject_data.append(reject_str)

        return chosen_data + reject_data
