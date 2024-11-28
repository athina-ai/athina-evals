# Guardrails CorrectLanguage Evaluator
# https://hub.guardrailsai.com/validator/scb-10x/correct_language

import time
from typing import List, Optional, Dict
from athina.interfaces.result import EvalResult, EvalResultMetric
from athina.helpers.logger import logger
from ...base_evaluator import BaseEvaluator
from athina.metrics.metric_type import MetricType


# Passes when the text matched the specified language, fails when the text doesn't match the specified language.
class CorrectLanguage(BaseEvaluator):
    _expected_language_iso: str
    _threshold: float

    def __init__(
        self,
        expected_language_iso: str = "en",
        threshold: float = 0.75,
    ):
        from guardrails.hub import CorrectLanguage as GuardrailsCorrectLanguage

        self._expected_language_iso = expected_language_iso
        self._threshold = threshold

        # Initialize Validator
        self.validator = GuardrailsCorrectLanguage(
            expected_language_iso=self._expected_language_iso,
            threshold=self._threshold,
            on_fail="noop",
        )

    @property
    def name(self) -> str:
        return "CorrectLanguage"

    @property
    def display_name(self) -> str:
        return "Correct Language"

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
                    "Text doesn't match the specified language"
                    if validation_passed
                    else "Text matched the specified language"
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
