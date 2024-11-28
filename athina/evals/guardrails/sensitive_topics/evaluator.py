# Guardrails Sensitive topics Evaluator
# https://hub.guardrailsai.com/validator/guardrails/sensitive_topics

import os
import time
from typing import List, Optional, Dict
from athina.interfaces.result import EvalResult, EvalResultMetric
from athina.helpers.logger import logger
from athina.errors.exceptions import NoOpenAiApiKeyException
from athina.keys import OpenAiApiKey
from ...base_evaluator import BaseEvaluator
from athina.metrics.metric_type import MetricType

# Passes when the text doesn't contain any sensitive topics, fails when the text contains.
class ContainsNoSensitiveTopics(BaseEvaluator):
    _sensitive_topics: List[str]
    _default_sensitive_topics = ["adult content", "hate speech", "illegal activities", "politics", "violence"]

    def __init__(
        self,
        sensitive_topics: List[str] = _default_sensitive_topics, 
        open_ai_api_key: Optional[str] = None
    ):
        from guardrails.hub import SensitiveTopic
        if open_ai_api_key is None:
            if OpenAiApiKey.get_key() is None:
                raise NoOpenAiApiKeyException()
            os.environ['OPENAI_API_KEY'] = OpenAiApiKey.get_key()
        else:
            self.open_ai_api_key = open_ai_api_key
        # Initialize Validator
        self.validator = SensitiveTopic(
            sensitive_topics=sensitive_topics,
            disable_classifier=True,
            disable_llm=False,
            on_fail="exception",
        )

    @property
    def name(self) -> str:
        return "ContainsNoSensitiveTopics"

    @property
    def display_name(self) -> str:
        return "Contains No Sensitive Topics"

    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.PASSED.value]

    @property
    def required_args(self) -> List[str]:
        return ["response"]  # TODO: allow running this on user_query OR response

    @property
    def examples(self):
        pass

    def to_config(self) -> Optional[Dict]:
        return None

    def is_failure(self, result: bool) -> bool:
        return not(bool(result))

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
                grade_reason = "Text doesn't contain sensitive topics" if validation_passed else "Text contains sensitive topics"
            except Exception as e:
                validation_passed = False
                grade_reason = str(e).replace('Validation failed for field with errors:', '')

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