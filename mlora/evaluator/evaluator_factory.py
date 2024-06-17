from mlora.model.llm.model_llm import LLMModel
from mlora.model.tokenizer.tokenizer import Tokenizer
from mlora.evaluator.evaluator import Evaluator
from mlora.evaluator.mmlu_evaluator import MMLUEvaluator


class EvaluatorFactory():
    @staticmethod
    def create(model: LLMModel,
               tokenizer: Tokenizer,
               evaluator_type: str,
               data: str) -> Evaluator:
        type_args = evaluator_type.split(":")

        assert len(type_args) >= 1, f"error args {type_args}"

        # the first is the evaluator_class
        evaluator_class = type_args[0]

        if evaluator_class == "mmlu":
            return MMLUEvaluator(model, tokenizer, data, type_args[1:])
        else:
            raise f"Not support: {evaluator_class}"
