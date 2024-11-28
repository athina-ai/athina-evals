# Guardrails PolitenessCheck
# https://hub.guardrailsai.com/validator/guardrails/politeness_check

import os
import time
from typing import Dict, List, Optional
from athina.helpers.logger import logger
from ...base_evaluator import BaseEvaluator
from athina.metrics.metric_type import MetricType
from athina.interfaces.result import EvalResult, EvalResultMetric
from athina.errors.exceptions import NoOpenAiApiKeyException
from athina.keys import OpenAiApiKey


# Passes when the text is polite, fails when the text is not polite.
class PolitenessCheck(BaseEvaluator):
    _llm_callable: str

    def __init__(
        self, llm_callable: str = "gpt3.5-turbo", open_ai_api_key: Optional[str] = None
    ):
        from guardrails.hub import PolitenessCheck as GuardrailsPolitenessCheck

        open_ai_api_key = open_ai_api_key or OpenAiApiKey.get_key()
        if open_ai_api_key is None:
            raise NoOpenAiApiKeyException()
        os.environ["OPENAI_API_KEY"] = open_ai_api_key

        self._llm_callable = llm_callable
        # Initialize Validator
        self.validator = GuardrailsPolitenessCheck(
            llm_callable=self._llm_callable,
            on_fail="noop",
        )

    @property
    def name(self) -> str:
        return "PolitenessCheck"

    @property
    def display_name(self) -> str:
        return "Politeness Check"

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
            guard_result = guard.parse(text)
            grade_reason = (
                "Text is polite"
                if guard_result.validation_passed
                else "Text is not polite"
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
