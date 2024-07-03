import traceback
from abc import ABC, abstractmethod
import time
from typing import List, Optional
from athina.interfaces.result import EvalResult, EvalResultMetric
from athina.interfaces.athina import AthinaExperiment
from athina.interfaces.model import Model
from athina.llms.openai_service import OpenAiService
from athina.helpers.logger import logger
from athina.interfaces.data import DataPoint
from athina.services.athina_api_service import AthinaApiService
from athina.metrics.metric_type import MetricType
from athina.llms.abstract_llm_service import AbstractLlmService
from .example import FewShotExample
from ..base_evaluator import BaseEvaluator


class LlmEvaluator(BaseEvaluator):
    llm_service: AbstractLlmService
    _model: str
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
    You are an expert at evaluating responses by an AI.

    Based on the instructions provided, you will evaluate the response and determine if it passes or fails.
    
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
        system_message_template: Optional[str] = None,
        user_message_template: Optional[str] = None,
        llm_service: Optional[AbstractLlmService] = None,
    ):
        if llm_service is not None and isinstance(llm_service, AbstractLlmService):
            self.llm_service = llm_service
        else:
            self.llm_service = OpenAiService()
        if model is None:
            self._model = self.default_model
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


    @property
    @abstractmethod
    def default_model(self):
        """The default model for the evaluator."""
        pass

    def __str__(self):
        formatted_args = [str(value) for value in self.required_args]
        return f"Docstring: {self.__doc__}\nRequired Arguments: {formatted_args}"

    
    def _system_message(self) -> str:
        return self._system_message_template


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
    
        metrics = []
        try:
            result = chat_completion_response_json["result"]
            explanation = chat_completion_response_json["explanation"]
            failure = self.is_failure(result)
            passed_value = 1 - float(failure)
            metrics.append(EvalResultMetric(id=MetricType.PASSED.value, value=passed_value))

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
            metrics=metrics,
        )
        return {k: v for k, v in llm_eval_result.items() if v is not None}
    
