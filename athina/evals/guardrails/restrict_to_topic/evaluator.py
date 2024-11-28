# Guardrails Restrict To Topic
# https://hub.guardrailsai.com/validator/tryolabs/restricttotopic

import os
import time
from typing import List, Optional, Dict
from athina.interfaces.result import EvalResult, EvalResultMetric
from athina.helpers.logger import logger
from athina.errors.exceptions import NoOpenAiApiKeyException
from athina.keys import OpenAiApiKey
from ...base_evaluator import BaseEvaluator
from athina.metrics.metric_type import MetricType


# Passes when the text is restricted to the specified topics, fails when the text doesn't.
class RestrictToTopic(BaseEvaluator):
    _valid_topics: List[str]
    _invalid_topics = []

    def __init__(
        self,
        valid_topics: List[str],
        invalid_topics: List[str] = [],
        open_ai_api_key: Optional[str] = None,
    ):
        from guardrails.hub import RestrictToTopic

        if open_ai_api_key is None:
            if OpenAiApiKey.get_key() is None:
                raise NoOpenAiApiKeyException()
            os.environ["OPENAI_API_KEY"] = OpenAiApiKey.get_key()
        else:
            self.open_ai_api_key = open_ai_api_key
        self._valid_topics = valid_topics
        self._invalid_topics = invalid_topics

        # Initialize Validator
        self.validator = RestrictToTopic(
            valid_topics=self._valid_topics,
            invalid_topics=self._invalid_topics,
            disable_classifier=True,
            disable_llm=False,
            on_fail="noop",
        )

    @property
    def name(self) -> str:
        return "RestrictToTopic"

    @property
    def display_name(self) -> str:
        return "Restrict To Topic"

    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.PASSED.value]

    @property
    def required_args(self) -> List[str]:
        return ["response"]

    @property
    def examples(self):
        pass

    def to_config(self) -> Optional[Dict]:
        return None

    def is_failure(self, result: bool) -> bool:
        return not (bool(result))

    def _evaluate(self, **kwargs) -> EvalResult:
        from guardrails import Guard

        """
        Run the Guardrails evaluator.
        """
        start_time = time.time()
        self.validate_args(**kwargs)
        metrics = []
        try:
            text = kwargs["response"]
            # Setup Guard
            guard = Guard.from_string(validators=[self.validator])
            validation_passed = False
            # Pass LLM output through guard
            try:
                guard_result = guard.parse(text)
                validation_passed = guard_result.validation_passed
                grade_reason = (
                    "Text is restricted to the specified topics"
                    if validation_passed
                    else "Text is not restricted to the specified topics"
                )
            except Exception as e:
                validation_passed = False
                grade_reason = str(e).replace(
                    "Validation failed for field with errors:", ""
                )

            # Boolean evaluator
            metrics.append(
                EvalResultMetric(
                    id=MetricType.PASSED.value,
                    value=float(validation_passed),
                )
            )
        except Exception as e:
            logger.error(f"Error occurred during eval: {e}")
            raise e

        end_time = time.time()
        eval_runtime_ms = int((end_time - start_time) * 1000)
        llm_eval_result = EvalResult(
            name=self.name,
            display_name=self.display_name,
            data=kwargs,
            failure=self.is_failure(validation_passed),
            reason=grade_reason,
            runtime=eval_runtime_ms,
            model="gpt-3.5-turbo",
            metrics=metrics,
        )
        return {k: v for k, v in llm_eval_result.items() if v is not None}
