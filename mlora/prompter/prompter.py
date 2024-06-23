import yaml
import jinja2
from typing import Dict, List
from jinja2.sandbox import ImmutableSandboxedEnvironment
from abc import abstractmethod


class Prompter:
    template_: jinja2.Template = None

    def __init__(self, template: str):
        with open(template) as fp:
            template_str = yaml.safe_load(fp)
        jinja_env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True)

        self.template_ = jinja_env.from_string(template_str["template"])

    @abstractmethod
    def generate_prompt(self, data_points: List[Dict[str, str]]) -> List[str]:
        ...
