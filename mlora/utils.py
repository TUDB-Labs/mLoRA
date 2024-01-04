import torch
import json
import os.path as osp
from typing import Union
from transformers import LlamaForCausalLM


# convert huggingface model to pytorch model
def convert_hf_to_pth(source: str, dest: str):
    src_model = LlamaForCausalLM.from_pretrained(source)
    # src_model.eval()
    torch.save(src_model.state_dict(), dest)


# manage templates and prompt building.
class Prompter:
    def __init__(self, file_name: str, verbose: bool = False):
        self._verbose = verbose
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {file_name}: {self.template['description']}"
            )

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
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()
