from typing import List, Optional, Dict, Any
from ..llm_evaluator import LlmEvaluator
from athina.evals.eval_type import LlmEvalTypeId
from ..example import FewShotExample


class CustomLlmEval(LlmEvaluator):
    """
    This evaluator can be configured with custom examples and instructions.
    """

    _display_name: str = None
    _metric_ids: List[str] = None
    _model: str = None
    _required_args: Dict[str, Any] = None
    _examples: List[FewShotExample] = None
    _system_message_template: Optional[str] = None
    _user_message_template: Optional[str] = None

    def __init__(
        self,
        display_name: str = None,
        metric_ids: List[str] = None,
        model: str = None,
        required_args: Dict[str, Any] = {},
        examples: List[FewShotExample] = [],
        system_message_template: Optional[str] = None,
        user_message_template: Optional[str] = None,
        **kwargs
    ):
        if display_name is None:
            raise ValueError("display_name is not defined")
        if model is None:
            raise ValueError("model is not defined")

        self._display_name = display_name
        self._metric_ids = metric_ids
        self._model = model
        self._required_args = required_args
        self._examples = examples

        if system_message_template is not None:
            self._system_message_template = system_message_template

        if user_message_template is not None:
            self._user_message_template = user_message_template

        super().__init__(
            model=self._model,
            system_message_template=self._system_message_template,
            user_message_template=self._user_message_template,
            **kwargs
        )

    @property
    def name(self):
        return LlmEvalTypeId.CUSTOM.value
        
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
        
