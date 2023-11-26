from typing import List
from ..llm_evaluator import LlmEvaluator


class CustomGrader(LlmEvaluator):
    """
    This evaluator checks if the response is correct according to a provided `grading_criteria`.
    """

    NAME = "custom_grader"
    DISPLAY_NAME = "Custom"
    DEFAULT_MODEL = "gpt-4"
    REQUIRED_ARGS: List[str] = ["response"]
    EXAMPLES = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if "grading_criteria" not in kwargs:
            raise Exception("grading_criteria is required for CustomLlmEvaluator")
