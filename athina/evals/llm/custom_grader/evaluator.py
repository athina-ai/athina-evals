from typing import List
from ..llm_evaluator import LlmEvaluator
from ..eval_type import AthinaEvalTypeId


class CustomGrader(LlmEvaluator):
    """
    This evaluator checks if the response is correct according to a provided `grading_criteria`.
    """

    _examples = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "grading_criteria" not in kwargs:
            raise Exception("grading_criteria is required for CustomLlmEvaluator")
        if "examples" in kwargs:
            self._examples = kwargs["examples"]

    def name(self):
        return AthinaEvalTypeId.CUSTOM.value

    def metric_id(self) -> str:
        return None

    def display_name(self):
        return "Custom"

    def default_model(self):
        return "gpt-4-1106-preview"

    def required_args(self):
        return ["response"]

    def examples(self):
        return self._examples
