from typing import Mapping, Type

from mlora.config import DatasetConfig

from .contrastive_instruction_data_prompter import ContrastiveInstructionDataPrompter
from .instruction_data_prompter import InstructionDataPrompter
from .ppo_data_prompter import PPODataPrompter
from .preference_data_prompter import PreferenceDataPrompter
from .prompter import Prompter

_PROMPTER_CLASS: Mapping[str, Type[Prompter]] = {
    "instruction": InstructionDataPrompter,
    "preference": PreferenceDataPrompter,
    "contrastive_instruction": ContrastiveInstructionDataPrompter,
    "ppo": PPODataPrompter,
}


class PrompterFactory:
    @staticmethod
    def create(config: DatasetConfig) -> Prompter:
        assert config.prompt_type_ in _PROMPTER_CLASS
        return _PROMPTER_CLASS[config.prompt_type_](config.prompt_path_)


__all__ = [
    "PrompterFactory",
    "Prompter",
    "PreferenceDataPrompter",
    "InstructionDataPrompter",
    "ContrastiveInstructionDataPrompter",
    "PPODataPrompter",
]
