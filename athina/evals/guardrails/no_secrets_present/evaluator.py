# Guardrails No Secrets Present
# https://hub.guardrailsai.com/validator/guardrails/secrets_present

import time
from typing import Dict, List, Optional
from athina.helpers.logger import logger
from ...base_evaluator import BaseEvaluator
from athina.metrics.metric_type import MetricType
from athina.interfaces.result import EvalResult, EvalResultMetric


# Passes when the text has no secrets, fails when the text contains secrets.
class NoSecretsPresent(BaseEvaluator):

    def __init__(
        self,
    ):
        from guardrails.hub import SecretsPresent as GuardrailsSecretsPresent

        # Initialize Validator
        self.validator = GuardrailsSecretsPresent(
            on_fail="noop",
        )

    @property
    def name(self) -> str:
        return "NoSecretsPresent"

    @property
    def display_name(self) -> str:
        return "No Secrets Present"

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
        """
        Run the Guardrails evaluator.
        """
        from guardrails import Guard

        start_time = time.time()
        self.validate_args(**kwargs)
        metrics = []
        try:
            text = kwargs["response"]
            # Setup Guard
            guard = Guard.from_string(validators=[self.validator])
            # Pass LLM output through guard
            guard_result = guard.parse(text)
            grade_reason = (
                "Text contains no secrets"
                if guard_result.validation_passed
                else "Text has secrets"
            )
            # Boolean evaluator
            metrics.append(
                EvalResultMetric(
                    id=MetricType.PASSED.value,
                    value=float(guard_result.validation_passed),
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
            failure=self.is_failure(guard_result.validation_passed),
            reason=grade_reason,
            runtime=eval_runtime_ms,
            model=None,
            metrics=metrics,
        )
        return {k: v for k, v in llm_eval_result.items() if v is not None}
