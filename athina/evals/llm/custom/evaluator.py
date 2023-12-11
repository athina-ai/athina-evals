from typing import List, Optional
from ..llm_evaluator import LlmEvaluator
from ..eval_type import AthinaEvalTypeId
from ..example import FewShotExample


class CustomLlmEval(LlmEvaluator):
    """
    This evaluator can be configured with custom examples and instructions.
    """

    _display_name: str = None
    _metric_id: str = None
    _model: str = None
    _required_args: List[str] = None
    _examples: List[FewShotExample] = None
    _system_message_template: Optional[str] = None
    _user_message_template: Optional[str] = None

    def __init__(
        self,
        display_name: str = None,
        metric_id: str = None,
        model: str = None,
        required_args: List[str] = [],
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
        self._metric_id = metric_id
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

    def name(self):
        return AthinaEvalTypeId.CUSTOM.value

    def metric_id(self) -> str:
        return self._metric_id

    def display_name(self):
        return self._display_name

    def default_model(self):
        return self._model

    def required_args(self):
        return self._required_args

    def examples(self):
        return self._examples
