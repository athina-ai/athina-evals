from typing import List
from ..llm_evaluator import LlmEvaluator
from .examples import DOES_RESPONSE_ANSWER_QUERY_EVAL_EXAMPLES


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
        2. Make sure to also consider these instructions: {additional_instructions}
        3. Determine if the response answers specifically what the user is asking about, and covers all aspects of the user's query.
        4. Provide a brief explanation of why the response does or does not answer the user's query sufficiently, labeled as 'explanation', leading up to a verdict (Pass/Fail) labeled as 'result'.
        5. Return a JSON object in the following format: "result": 'result', "explanation": 'explanation'.

        ### EXAMPLES ###
        Here's are some examples: 
        {examples}

        Now consider the following:
        user's query: {query}.
        response: {response}.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def name(self):
        return "does_response_answer_query"

    def display_name(self):
        return "Does Response Answer Query"

    def default_model(self):
        return "gpt-4-1106-preview"

    def required_args(self):
        return ["query", "response"]

    def examples(self):
        return DOES_RESPONSE_ANSWER_QUERY_EVAL_EXAMPLES

    def _user_message(
        self,
        query: str,
        response: str,
        additional_instructions: str = "",
        **kwargs,
    ) -> str:
        return self.USER_MESSAGE_TEMPLATE.format(
            query=query,
            response=response,
            additional_instructions=additional_instructions,
            examples=self._examples_str(),
        )
