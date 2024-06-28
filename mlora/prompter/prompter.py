from abc import abstractmethod
from typing import Dict, List

import jinja2
import yaml
from jinja2.sandbox import ImmutableSandboxedEnvironment


class Prompter:
    template_: jinja2.Template

    def __init__(self, template: str):
        with open(template) as fp:
            template_str = yaml.safe_load(fp)
        jinja_env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

        self.template_ = jinja_env.from_string(template_str["template"])

    @abstractmethod
    def generate_prompt(self, data_points: List[Dict[str, str]]) -> List[str]: ...
