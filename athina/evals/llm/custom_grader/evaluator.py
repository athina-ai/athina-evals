from typing import List
from ..llm_evaluator import LlmEvaluator
from athina.interfaces.eval_type import LlmEvalTypeId


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

    @property
    def name(self):
        return LlmEvalTypeId.CUSTOM.value

    @property
    def metric_id(self) -> str:
        return None

    @property
    def display_name(self):
        return "Custom"

    @property
    def default_model(self):
        return "gpt-4-1106-preview"

    @property
    def required_args(self):
        return ["response"]

    @property
    def examples(self):
        return self._examples
