from typing import List, Optional, Dict, Any
from ..llm_evaluator import LlmEvaluator
from athina.evals.eval_type import LlmEvalTypeId
from ..example import FewShotExample


class CustomPrompt(LlmEvaluator):
    """
    This evaluator can be configured with custom examples and instructions.
    """

    _eval_prompt: Optional[str] = None
    _display_name: str = None
    _metric_ids: List[str] = None
    _model: str = None
    _required_args: List[str] = None
    _examples: List[FewShotExample] = None

    def __init__(
        self,
        eval_prompt: str,
        display_name: str = None,
        metric_ids: List[str] = None,
        model: str = None,
        required_args: List[str] = [],
        examples: List[FewShotExample] = [],
        **kwargs
    ):
        if eval_prompt is None:
            raise ValueError("eval_prompt is not defined")
        if model is None:
            raise ValueError("model is not defined")


        self._eval_prompt = eval_prompt
        self._display_name = display_name
        self._metric_ids = metric_ids
        self._model = model
        self._required_args = required_args
        self._examples = examples
        self._system_message_template = self.DEFAULT_SYSTEM_MESSAGE_TEMPLATE + self.RETURN_FORMAT_INSTRUCTIONS

        super().__init__(
            model=self._model,
            system_message_template=self._system_message_template,
            user_message_template=self._eval_prompt,
            **kwargs
        )

    @property
    def name(self):
        return LlmEvalTypeId.CUSTOM_PROMPT.value
        
    @property
    def metric_ids(self) -> str:
        return self._metric_ids
        
    @property
    def display_name(self):
        return self._display_name
        
    @property
    def default_model(self):
        return self._model
        
    @property
    def required_args(self):
        return self._required_args
        
    @property
    def examples(self):
        return self._examples
        
