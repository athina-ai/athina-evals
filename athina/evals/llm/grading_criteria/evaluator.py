from typing import List, Optional
from ..llm_evaluator import LlmEvaluator
from ..eval_type import AthinaEvalTypeId


class GradingCriteria(LlmEvaluator):
    """
    This evaluator checks if the response is correct according to a provided `grading_criteria`.
    """

    _examples = []

    def __init__(self, 
        grading_criteria: str, 
        model: Optional[str] = None,
    ):
        if grading_criteria is None:
            raise Exception("Eval is incorrectly configured: grading_criteria is required for GradingCriteria evaluator")
        super().__init__(
            model=model,
            grading_criteria=grading_criteria
        )

    @property
    def name(self):
        return AthinaEvalTypeId.GRADING_CRITERIA.value

    @property
    def metric_ids(self) -> str:
        return []

    @property
    def display_name(self):
        return "Response matches Grading Criteria"

    @property
    def default_model(self):
        return "gpt-4-1106-preview"

    @property
    def required_args(self):
        return ["response"]

    @property
    def examples(self):
        return self._examples
