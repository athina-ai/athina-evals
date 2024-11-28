from typing import List, Optional
from ..llm_evaluator import LlmEvaluator
from .examples import FAITHFULNESS_EVAL_EXAMPLES
from athina.evals.eval_type import LlmEvalTypeId
from athina.metrics.metric_type import MetricType


class Faithfulness(LlmEvaluator):
    """
    This evaluator checks if the response can be inferred using the information provided as context.
    """

    SYSTEM_MESSAGE_TEMPLATE = """ 
    You are an expert at evaluating whether the response can be inferred using ONLY the information provided as context and chat history. If chat history is not provided, consider only the context.
    You are not concerned with factual correctness or accuracy. You are only determining whether the response can be inferred directly from the information provided as context and chat history.
    """

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step.
        1. Consider the following: 
        context: {context}.
        chat history: {chat_history}
        response: {response}.
        2. Determine if the response can be inferred using ONLY the information provided in the context and chat history.
        3. If the chat history is not provided, consider only the context.
        4. Provide a brief explanation of why the response can or cannot be inferred purely from the context and chat history, labeled as 'explanation', leading up to a verdict (Pass/Fail) labeled as 'result'.
        5. Return a JSON object in the following format: "result": 'result', "explanation": 'explanation'.

        ### EXAMPLES ###
        Here are some examples: 
        {examples}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return LlmEvalTypeId.FAITHFULNESS.value

    @property
    def display_name(self):
        return "Faithfulness"

    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.PASSED.value]

    @property
    def default_model(self):
        return "gpt-4-1106-preview"

    @property
    def required_args(self):
        return ["context", "response"]

    @property
    def examples(self):
        return FAITHFULNESS_EVAL_EXAMPLES

    def is_failure(self, result) -> Optional[bool]:
        return bool(str(result).lower() == "fail")

    def _user_message(
        self,
        context: str,
        response: str,
        **kwargs,
    ) -> str:
        """
        Generates data for evaluation.

        :param context: list of strings of retrieved context
        :param response: llm response
        :return: A dictionary with formatted data for evaluation
        """
        joined_context = "\n".join(context)
        # Check if chat_history is provided and format it
        chat_history = kwargs.get("chat_history", [])
        formatted_chat_history = (
            "\n".join(chat_history) if chat_history else "No chat history provided."
        )

        return self.USER_MESSAGE_TEMPLATE.format(
            context=joined_context,
            response=response,
            chat_history=formatted_chat_history,
            examples=self.examples,
        )
