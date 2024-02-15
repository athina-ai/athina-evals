# Guardrails Evaluator
# https://hub.guardrailsai.com/validator/guardrails/gibberish_text

import math
from abc import abstractmethod
from typing import Optional, List
from athina.interfaces.athina import AthinaExperiment
from athina.interfaces.model import Model
import time
from typing import Optional, Any
from athina.interfaces.result import EvalResult, EvalResultMetric
from athina.interfaces.model import Model
from athina.helpers.logger import logger
from ...base_evaluator import BaseEvaluator
from datasets import Dataset
from athina.keys import OpenAiApiKey
from guardrails.hub import GibberishText
from guardrails import Guard


# Passes when the text is sensible, fails when the text is gibberish.
class GibberishTextEvaluator(BaseEvaluator):
    _validation_method: str
    _threshold: float

    def __init__(self, validation_method: str = "sentence", threshold: float = 0.75):
        self._validation_method = validation_method
        self._threshold = threshold

    @property
    def name(self) -> str:
        return "GibberishText"

    @property
    def display_name(self) -> str:
        return "Gibberish Text"

    @property
    def metric_ids(self) -> List[str]:
        return ["SensibleText"]

    @property
    def required_args(self) -> List[str]:
        return ["response"]  # TODO: allow running this on user_query OR response

    @property
    def examples(self):
        pass

    def _evaluate(self, **kwargs) -> EvalResult:
        """
        Run the Guardrails evaluator.
        """
        start_time = time.time()
        self.validate_args(**kwargs)
        metrics = []
        try:

            text = kwargs["response"]

            # Initialize Validator
            val = GibberishText(
                threshold=self._threshold,
                validation_method=self._validation_method,
                on_fail="noop",
            )

            # Setup Guard
            guard = Guard.from_string(validators=[val])

            # Pass LLM output through guard
            guard_result = guard.parse(text)
            passed = guard_result.validation_passed
            grade_reason = "Text is sensible" if passed else "Text is gibberish"

            # Boolean evaluator
            metrics.append(
                EvalResultMetric(
                    id="SensibleText", value=guard_result.validation_passed
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
            failure=None,
            reason=grade_reason,
            runtime=eval_runtime_ms,
            model=None,
            metrics=metrics,
        )
        return {k: v for k, v in llm_eval_result.items() if v is not None}
