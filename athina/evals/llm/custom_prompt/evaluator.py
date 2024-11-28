import json
import time
from athina.helpers.logger import logger
from typing import List, Optional, Dict
from jinja2 import Environment
from athina.helpers.jinja_helper import PreserveUndefined

from athina.llms.abstract_llm_service import AbstractLlmService
from ..llm_evaluator import LlmEvaluator
from athina.evals.eval_type import LlmEvalTypeId
from ..example import FewShotExample
from athina.interfaces.result import EvalResult, EvalResultMetric
from athina.metrics.metric_type import MetricType


class CustomPrompt(LlmEvaluator):
    """
    This evaluator can be configured with custom examples and instructions.
    """

    _eval_prompt: Optional[str] = None
    _output_type: Optional[str] = None
    _display_name: str = None
    _metric_ids: List[str] = None
    _model: str = None
    _required_args: List[str] = None
    _examples: List[FewShotExample] = None

    def __init__(
        self,
        eval_prompt: str,
        output_type: str = "boolean",
        display_name: str = None,
        metric_ids: List[str] = None,
        model: str = None,
        required_args: List[str] = [],
        examples: List[FewShotExample] = [],
        llm_service: Optional[AbstractLlmService] = None,
        **kwargs,
    ):
        if eval_prompt is None:
            raise ValueError("eval_prompt is not defined")
        if model is None:
            raise ValueError("model is not defined")

        self._eval_prompt = eval_prompt
        self._output_type = output_type
        self._display_name = display_name
        self._metric_ids = metric_ids
        self._model = model
        self._required_args = required_args
        self._examples = examples
        self._system_message_template = None

        prompt_messages = kwargs.get("prompt_messages", [])
        if (
            prompt_messages
            and len(prompt_messages) > 0
            and prompt_messages[0].get("role") == "system"
            and prompt_messages[0].get("content")
            and prompt_messages[0]["content"].strip()
        ):
            self._system_message_template = prompt_messages[0]["content"]

        super().__init__(
            model=self._model,
            system_message_template=self._system_message_template,
            user_message_template=self._eval_prompt,
            llm_service=llm_service,
            **kwargs,
        )
        # Create a custom Jinja2 environment with double curly brace delimiters and PreserveUndefined
        self.env = Environment(
            variable_start_string="{{",
            variable_end_string="}}",
            undefined=PreserveUndefined,
        )

    @property
    def name(self):
        return LlmEvalTypeId.CUSTOM_PROMPT.value

    @property
    def metric_ids(self) -> List[str]:
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

    def to_config(self) -> Optional[Dict]:
        return {
            "eval_prompt": self._eval_prompt,
        }

    def is_failure(self, result) -> Optional[bool]:
        return bool(str(result).lower() == "fail")

    def _user_message(self, **kwargs) -> str:
        if "chat_history" in kwargs:
            kwargs["chat_history"] = json.dumps(kwargs["chat_history"], indent=2)
        template = self.env.from_string(self._user_message_template)
        return template.render(**kwargs)

    def _system_message(self) -> str:
        if self._system_message_template:
            return self._system_message_template
        else:
            if self._output_type == "boolean":
                return (
                    "### INSTRUCTIONS ###\n"
                    "You are an expert at evaluating responses by an AI.\n"
                    "Based on the instructions provided, you will evaluate the response and determine if it passes or fails.\n"
                    "You MUST return a JSON object with the following fields:\n"
                    "- result: Result must be either 'Pass' or 'Fail'.\n"
                    "- explanation: An explanation of why the result is Pass or Fail.\n"
                )
            elif self._output_type == "numeric":
                return (
                    "### INSTRUCTIONS ###\n"
                    "You are an expert at evaluating responses by an AI.\n"
                    "Based on the instructions provided, you will evaluate the response and provide a score.\n"
                    "You MUST return a JSON object with the following fields:\n"
                    "- score: The score based on the provided grading criteria.\n"
                    "- explanation: An explanation of the score.\n"
                )

    def _evaluate(self, **kwargs) -> EvalResult:
        """
        Run the LLM evaluator.
        """

        start_time = time.time()
        # Validate that correct args were passed
        self.validate_args(**kwargs)

        # Construct Prompt
        messages = self._prompt_messages(**kwargs)

        metrics = []
        try:
            # Run the LLM Completion
            chat_completion_response_json: dict = self.llm_service.json_completion(
                model=self._model,
                messages=messages,
                temperature=self.TEMPERATURE,
            )

            if self._output_type == "boolean":
                result = chat_completion_response_json["result"]
                explanation = chat_completion_response_json["explanation"]
                failure = self.is_failure(result)
                passed_value = 1 - float(failure)
                metrics.append(
                    EvalResultMetric(id=MetricType.PASSED.value, value=passed_value)
                )
            elif self._output_type == "numeric":
                score = chat_completion_response_json["score"]
                explanation = chat_completion_response_json["explanation"]
                metrics.append(EvalResultMetric(id=MetricType.SCORE.value, value=score))
                failure = None  # Numeric evaluations don't have a pass/fail result

        except Exception as e:
            logger.error(f"Error occurred during eval: {e}")
            if isinstance(e, (ValueError, KeyError)):
                raise ValueError(
                    "LLM evals must return a result/score and explanation. The LLM response did not return the correct structure for parsing evaluation results."
                )
            else:
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
