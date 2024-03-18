from typing import List, Optional
from ..llm_evaluator import LlmEvaluator
from .examples import CONTEXT_CONTAINS_ENOUGH_INFORMATION_EXAMPLES
from athina.evals.eval_type import LlmEvalTypeId
from athina.metrics.metric_type import MetricType


class ContextContainsEnoughInformation(LlmEvaluator):
    """
    This evaluator checks if the user's query can be answered using only the information in the context.
    """

    SYSTEM_MESSAGE_TEMPLATE = """
    You are an expert at evaluating whether a chatbot can answer a user's query using ONLY the information provided to you as context.
    You are not concerned with factual correctness or accuracy. You only care whether the context contains enough information to answer the user's query.
    """

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step:

        1. Consider the following: 
        user's query: {query}.
        context: {context}.
        2. Determine if the chatbot can answer the user's query with nothing but the "context" information provided to you.
        3. Provide a brief explanation of why the context does or does not contain sufficient information, labeled as 'explanation', leading up to a verdict (Pass/Fail) labeled as 'result'.
        4. Always return a JSON object in the following format: "result": 'result', "explanation": 'explanation'.

        Here's are some examples: 
        {examples}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return LlmEvalTypeId.CONTEXT_CONTAINS_ENOUGH_INFORMATION.value

    @property
    def display_name(self):
        return "Context Contains Enough Information"

    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.PASSED.value]

    @property
    def default_model(self):
        return "gpt-4-1106-preview"

    @property
    def required_args(self):
        return ["query", "context"]

    @property
    def examples(self):
        return CONTEXT_CONTAINS_ENOUGH_INFORMATION_EXAMPLES

    def is_failure(self, result) -> Optional[bool]:
        return bool(result == "Fail")

    def _user_message(
        self,
        query: str,
        context: str,
        **kwargs,
    ) -> str:
        """
        Generates data for evaluation.

        :param query: user query
        :param context: list of strings of retrieved context
        :return: A dictionary with formatted data for evaluation
        """
        joined_context = "\n".join(context)
        return self.USER_MESSAGE_TEMPLATE.format(
            query=query,
            context=joined_context,
            examples=self.examples,
        )
