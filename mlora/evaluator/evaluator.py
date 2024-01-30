from mlora.model.model import LLMModel
from mlora.tokenizer.tokenizer import Tokenizer

from abc import ABCMeta, abstractclassmethod


class Evaluator(metaclass=ABCMeta):
    model_: LLMModel = None
    tokenizer_: Tokenizer = None

    @abstractclassmethod
    def evaluate(self) -> float:
        ...
