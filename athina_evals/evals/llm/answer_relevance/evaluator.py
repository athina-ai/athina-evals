from typing import List
from ..base_llm_evaluator import BaseLlmEvaluator
from .examples import ANSWER_RELEVANCE_EVAL_EXAMPLES


class AnswerRelevanceEvaluator(BaseLlmEvaluator):
    """
    This evaluator checks if the response answers specifically what the user is asking about, and covers all aspects of the user's query.
    """

    REQUIRED_ARGS: List[str] = ["user_query", "response"]
    EXAMPLES = ANSWER_RELEVANCE_EVAL_EXAMPLES

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step.
        1. Consider the following: 
        user's query: {user_query}.
        response: {response}.
        2. Make sure to also consider these instructions: {additional_instructions}
        3. Determine if the response answers specifically what the user is asking about, and covers all aspects of the user's query.
        4. Provide a brief explanation of why the response does or does not answer the user's query sufficiently, labeled as 'explanation', leading up to a verdict (Pass/Fail) labeled as 'result'.
        5. Return a JSON object in the following format: "result": 'result', "explanation": 'explanation'.

        ### EXAMPLES ###
        Here's are some examples: 
        {examples}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _user_message(
        self,
        user_query: str,
        response: str,
        additional_instructions: str = "",
    ) -> str:
        return self.USER_MESSAGE_TEMPLATE.format(
            user_query=user_query,
            response=response,
            additional_instructions=additional_instructions,
            examples=self._examples_str(),
        )
