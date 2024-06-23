from mlora.config import DatasetConfig

from .prompter import Prompter
from .preference_data_prompter import PreferenceDataPrompter
from .instruction_data_prompter import InstructionDataPrompter


_PROMPTER_CLASS = {
    "instruction": InstructionDataPrompter,
    "preference": PreferenceDataPrompter
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
]
