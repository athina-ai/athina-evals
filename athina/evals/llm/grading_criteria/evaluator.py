from typing import List, Optional

from athina.llms.abstract_llm_service import AbstractLlmService
from ..llm_evaluator import LlmEvaluator
from athina.metrics.metric_type import MetricType
from athina.evals.eval_type import LlmEvalTypeId


class GradingCriteria(LlmEvaluator):
    """
    This evaluator checks if the response is correct according to a provided `grading_criteria`.
    """

    _examples = []

    def __init__(self, 
        grading_criteria: str, 
        model: Optional[str] = None,
        llm_service: Optional[AbstractLlmService] = None
    ):
        if grading_criteria is None:
            raise Exception("Eval is incorrectly configured: grading_criteria is required for GradingCriteria evaluator")
        super().__init__(
            model=model,
            grading_criteria=grading_criteria,
            llm_service=llm_service
        )

    @property
    def name(self):
        return LlmEvalTypeId.GRADING_CRITERIA.value

    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.PASSED.value]

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
    
    def is_failure(self, result) -> Optional[bool]:
        return bool(result == "Fail") 
