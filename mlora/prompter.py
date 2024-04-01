import json
import logging
import os.path as osp
from typing import Optional, Union, Dict

default_prompt_template = {
    "description": "Default Prompt Template Provided by m-LoRA",
    "prompt_input": "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Output:\n",
    "prompt_no_input": "### Instruction:\n{instruction}\n\n### Output:\n",
    "response_split": "### Output:"
}


# manage templates and prompt building.
class Prompter:
    def __init__(self, template: Optional[Union[Dict, str]] = None):
        if template is None:
            self.template = default_prompt_template
        elif isinstance(template, str):
            if not osp.exists(template):
                raise ValueError(f"Can't read {template}")
            with open(template) as fp:
                self.template = json.load(fp)
        else:
            self.template = template

        logging.info(
            f"Using prompt template: {self.template['description']}")

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}\n"
        logging.debug(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[-1].strip()
