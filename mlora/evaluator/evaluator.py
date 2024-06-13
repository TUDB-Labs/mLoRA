from mlora.model.llm.model import LLMModel
from mlora.model.tokenizer.tokenizer import Tokenizer

from abc import ABCMeta, abstractmethod


class Evaluator(metaclass=ABCMeta):
    model_: LLMModel = None
    tokenizer_: Tokenizer = None

    @abstractmethod
    def evaluate(self) -> float:
        ...
