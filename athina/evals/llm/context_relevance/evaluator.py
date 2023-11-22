from typing import List
from ..base_llm_evaluator import BaseLlmEvaluator
from .examples import CONTEXT_RELEVANCE_EVAL_EXAMPLES


class ContextRelevance(BaseLlmEvaluator):
    """
    This evaluator checks if the user's query can be answered using only the information in the context.
    """

    REQUIRED_ARGS: List[str] = ["user_query", "context"]
    EXAMPLES = CONTEXT_RELEVANCE_EVAL_EXAMPLES

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step.
        1. Consider the following: 
        user's query: {user_query}.
        context: {context}.
        2. Make sure to also consider these instructions: {additional_instructions}
        3. Determine if the user's query can be answered using only the information in the context.
        4. Provide a brief explanation of how the user's query can be answered using only the information in the context provided, labeled as 'explanation', leading up to a verdict (Pass/Fail) labeled as 'result'.
        5. The result must be Fail if the user's query cannot be answered using only the information in the context provided. Otherwise Pass.
        6. Return a JSON object in the following format: "result": 'result', "explanation": 'explanation'.

        ### EXAMPLES ###
        Here's are some examples: 
        {examples}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _user_message(
        self,
        user_query: str,
        context: str,
        additional_instructions: str = "",
    ) -> str:
        return self.USER_MESSAGE_TEMPLATE.format(
            user_query=user_query,
            context=context,
            additional_instructions=additional_instructions,
            examples=self._examples_str(),
        )
