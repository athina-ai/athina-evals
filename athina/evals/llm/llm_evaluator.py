import time
from typing import List, Optional
from athina.interfaces.result import EvalResult, EvalResultMetric
from athina.interfaces.athina import AthinaExperiment
from athina.interfaces.model import Model
from athina.llms.openai_service import OpenAiService
from athina.helpers.logger import logger
from athina.interfaces.data import DataPoint
from .example import FewShotExample
from ..base_evaluator import BaseEvaluator


class LlmEvaluator(BaseEvaluator):
    llm_service: OpenAiService
    grading_criteria: str
    _model: str
    _experiment: Optional[AthinaExperiment] = None
    _system_message_template: Optional[str] = None
    _user_message_template: Optional[str] = None

    TEMPERATURE = 0.0

    RETURN_FORMAT_INSTRUCTIONS = """
    You MUST return a JSON object with the following fields: 
    - result: Result must be either 'Pass' or 'Fail'.
    - explanation: An explanation of why the result is Pass or Fail.
    - score: (Optional) Use the scoring criteria specified.
    """

    DEFAULT_SYSTEM_MESSAGE_TEMPLATE = f""" 
    ### INSTRUCTIONS ###
    You are an expert at evaluating chatbot responses, according to some grading criteria.

    If it passes the grading criteria, then your result is Pass, otherwise it is Fail.
    
    """

    DEFAULT_USER_MESSAGE_TEMPLATE = """
    ### GRADING CRITERIA ###
    {grading_criteria}

    ### EXAMPLES ###
    {examples}

    ### RESPONSE TO EVALUATE ###
    {response}
    """

    EXAMPLES: FewShotExample = []

    def __init__(
        self,
        model: Optional[str] = None,
        grading_criteria: Optional[str] = None,
        system_message_template: Optional[str] = None,
        user_message_template: Optional[str] = None,
    ):
        self.llm_service = OpenAiService()
        self.grading_criteria = grading_criteria if grading_criteria else ""
        if model is None:
            self._model = self.default_model
        elif not Model.is_supported(model):
            raise ValueError(f"Unsupported model: {model}")
        else:
            self._model = model

        # Initialize message templates
        if system_message_template is None:
            self._system_message_template = (
                self.DEFAULT_SYSTEM_MESSAGE_TEMPLATE + self.RETURN_FORMAT_INSTRUCTIONS
            )
        else:
            self._system_message_template = system_message_template

        if user_message_template is None:
            self._user_message_template = self.DEFAULT_USER_MESSAGE_TEMPLATE
        else:
            self._user_message_template = user_message_template


    def __str__(self):
        formatted_args = {key: value.__name__ if hasattr(value, '__name__') else str(value)
                          for key, value in self.required_args.items()}
        return f"Docstring: {self.__doc__.strip()}\nRequired Arguments: {formatted_args}"
    
    def _system_message(self) -> str:
        return self._system_message_template

    def _user_message(self, **kwargs) -> str:
        return self._user_message_template.format(
            examples=self._examples_str(),
            grading_criteria=self.grading_criteria,
            **kwargs,
        )

    def _prompt_messages(self, **kwargs) -> List[dict]:
        return [
            {
                "role": "system",
                "content": self._system_message(),
            },
            {
                "role": "user",
                "content": self._user_message(**kwargs),
            },
        ]

    def _evaluate(self, **kwargs) -> EvalResult:
        """
        Run the LLM evaluator.
        """
        start_time = time.time()

        # Validate that correct args were passed
        self.validate_args(**kwargs)

        # Construct Prompt
        messages = self._prompt_messages(**kwargs)

        # Run the LLM Completion
        chat_completion_response_json: dict = self.llm_service.json_completion(
            model=self._model,
            messages=messages,
            temperature=self.TEMPERATURE,
        )

        try:
            metric = None
            result = chat_completion_response_json["result"]
            explanation = chat_completion_response_json["explanation"]
            failure = bool(result == "Fail")
            if "score" in chat_completion_response_json:
                score = chat_completion_response_json["score"]
                metric = EvalResultMetric(id=self.metric_id, value=score)
            else:
                metric = EvalResultMetric(id=self.metric_id, value=float(not failure))

        except Exception as e:
            logger.error(f"Error occurred during eval: {e}")
            raise e

        end_time = time.time()
        eval_runtime_ms = int((end_time - start_time) * 1000)
        llm_eval_result = EvalResult(
            name=self.name,
            display_name=self.display_name,
            data=kwargs,
            failure=failure,
            reason=explanation,
            runtime=eval_runtime_ms,
            model=self._model,
            metric=metric,
        )
        return {k: v for k, v in llm_eval_result.items() if v is not None}
