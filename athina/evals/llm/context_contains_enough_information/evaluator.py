from typing import List
from ..llm_evaluator import LlmEvaluator
from .examples import CONTEXT_CONTAINS_ENOUGH_INFORMATION_EXAMPLES


class ContextContainsEnoughInformation(LlmEvaluator):
    """
    This evaluator checks if the user's query can be answered using only the information in the context.
    """

    DEFAULT_MODEL = "gpt-4"
    REQUIRED_ARGS: List[str] = ["user_query", "context"]
    EXAMPLES = CONTEXT_CONTAINS_ENOUGH_INFORMATION_EXAMPLES

    SYSTEM_MESSAGE_TEMPLATE = f"""
    You are an expert at evaluating whether a chatbot can answer a user's query using ONLY the information provided to you as context.
    You are not concerned with factual correctness or accuracy. You only care whether the context contains enough information to answer the user's query.
    """

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step:

        1. Consider the following: 
        user's query: {user_query}.
        context: {context}.
        2. Make sure to also consider these instructions: {additional_instructions}
        3. Determine if the chatbot can answer the user's query with nothing but the "context" information provided to you.
        4. Provide a brief explanation of why the context does or does not contain sufficient information, labeled as 'explanation', leading up to a verdict (Pass/Fail) labeled as 'result'.
        5. Return a JSON object in the following format: "verdict": 'result', "explanation": 'explanation'.

        Here's are some examples: 
        {examples}
        """

    MY_TEMPLATE = """
        ### EXAMPLES ###
        Here's are some examples: 
        {examples}
    
        
        Let's think step by step.
        1. Consider the following: 
        user's query: {user_query}.
        context: {context}.
        2. Make sure to also consider these instructions: {additional_instructions}
        3. Determine if the user's query can be answered using only the information in the context.
        4. Provide a brief explanation of how the user's query can be answered using only the information in the context provided, labeled as 'explanation', leading up to a verdict (Pass/Fail) labeled as 'result'.
        5. The result must be Fail if the user's query cannot be answered using only the information in the context provided. Otherwise Pass.
        6. Return a JSON object in the following format: "result": 'result', "explanation": 'explanation'.
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
            examples=self.EXAMPLES,
        )
