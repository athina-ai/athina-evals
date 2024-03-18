from typing import List, Optional
from ..llm_evaluator import LlmEvaluator
from .examples import DOES_RESPONSE_ANSWER_QUERY_EVAL_EXAMPLES
from athina.evals.eval_type import LlmEvalTypeId
from athina.metrics.metric_type import MetricType


class DoesResponseAnswerQuery(LlmEvaluator):
    """
    This evaluator checks if the response answers specifically what the user is asking about, and covers all aspects of the user's query.
    """

    SYSTEM_MESSAGE_TEMPLATE = """
    You are an expert at evaluating whether the response answers specifically what the user is asking about, and covers all aspects of the user's query.
    You are not checking for correctness, or factual accuracy. You are only checking if the response answers the user's query.
    """

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step.
        1. Consider the following: 
        user's query: {query}.
        response: {response}.
        2. Determine if the response answers specifically what the user is asking about, and covers all aspects of the user's query.
        3. Provide a brief explanation of why the response does or does not answer the user's query sufficiently, labeled as 'explanation', leading up to a verdict (Pass/Fail) labeled as 'result'.
        4. Return a JSON object in the following format: "result": 'result', "explanation": 'explanation'

        ### EXAMPLES ###
        Here's are some examples: 
        {examples}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return LlmEvalTypeId.DOES_RESPONSE_ANSWER_QUERY.value

    @property
    def display_name(self):
        return "Does Response Answer Query"

    @property
    def default_model(self):
        return "gpt-4-1106-preview"

    @property
    def required_args(self):
        return ["query", "response"]

    @property
    def examples(self):
        return DOES_RESPONSE_ANSWER_QUERY_EVAL_EXAMPLES

    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.PASSED.value]

    def is_failure(self, result) -> Optional[bool]:
        return bool(result == "Fail") 

    def _user_message(
        self,
        query: str,
        response: str,
        **kwargs,
    ) -> str:
        """
        Generates data for evaluation.

        :param query: user query
        :param response: llm response
        :return: A dictionary with formatted data for evaluation
        """
        return self.USER_MESSAGE_TEMPLATE.format(
            query=query,
            response=response,
            examples=self._examples_str(),
        )
