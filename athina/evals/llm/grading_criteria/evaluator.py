from typing import List, Optional

from athina.llms.abstract_llm_service import AbstractLlmService
from ..llm_evaluator import LlmEvaluator
from athina.metrics.metric_type import MetricType
from athina.evals.eval_type import LlmEvalTypeId


class GradingCriteria(LlmEvaluator):
    """
    This evaluator checks if the response is correct according to a provided `grading_criteria`.
    """

    USER_MESSAGE_TEMPLATE = """
    ### GRADING CRITERIA ###
    {grading_criteria}

    ### EXAMPLES ###
    {examples}

    ### RESPONSE TO EVALUATE ###
    {response}
    """
    _examples = []
    grading_criteria = None
    def __init__(self, 
        grading_criteria: str, 
        model: Optional[str] = None,
        llm_service: Optional[AbstractLlmService] = None
    ):
        if grading_criteria is None:
            raise Exception("Eval is incorrectly configured: grading_criteria is required for GradingCriteria evaluator")
        self.grading_criteria = grading_criteria
        super().__init__(
            model=model,
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
    
    def to_config(self) -> Optional[dict]:
        return {
            "grading_criteria": self.grading_criteria
        }
    
    def is_failure(self, result) -> Optional[bool]:
        return bool(result == "Fail") 

    def _user_message(self, response, **kwargs) -> str:
        """
        Generates data for evaluation.

        :param response: llm response
        :return: A dictionary with formatted data for evaluation
        """
        return self.USER_MESSAGE_TEMPLATE.format(
            examples=self._examples_str(),
            grading_criteria=self.grading_criteria,
            response=response
        )