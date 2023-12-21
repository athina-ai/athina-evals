from typing import List
from ..llm_evaluator import LlmEvaluator
from .examples import FAITHFULNESS_EVAL_EXAMPLES
from athina.interfaces.eval_type import LlmEvalTypeId


class Faithfulness(LlmEvaluator):
    """
    This evaluator checks if the response can be inferred using the information provided as context.
    """

    SYSTEM_MESSAGE_TEMPLATE = """ 
    You are an expert at evaluating whether the response can be inferred using ONLY the information provided as context.
    You are not concerned with factual correctness or accuracy. You are only determining whether the response can be inferred directly from the information provided as context.
    """

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step.
        1. Consider the following: 
        context: {context}.
        response: {response}.
        2. Determine if the response can be inferred using ONLY the information provided in the context.
        3. Provide a brief explanation of why the response can or cannot be inferred purely from the context, labeled as 'explanation', leading up to a verdict (Pass/Fail) labeled as 'result'.
        4. Return a JSON object in the following format: "result": 'result', "explanation": 'explanation'.

        ### EXAMPLES ###
        Here's are some examples: 
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
    def metric_id(self) -> str:
        return None

    @property
    def default_model(self):
        return "gpt-4-1106-preview"

    @property
    def required_args(self):
        return ["context", "response"]

    @property
    def examples(self):
        return FAITHFULNESS_EVAL_EXAMPLES

    def _user_message(
        self,
        context: str,
        response: str,
        **kwargs,
    ) -> str:
        return self.USER_MESSAGE_TEMPLATE.format(
            context=context,
            response=response,
            examples=self._examples_str(),
        )
