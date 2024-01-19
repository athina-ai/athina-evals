import time
from typing import List

from athina.interfaces.result import EvalResult, EvalResultMetric
from athina.metrics.groundedness import GroundednessScore
from athina.helpers.logger import logger
from ....metrics.metric_type import MetricType
from ..llm_evaluator import LlmEvaluator
from .prompt import GROUNDEDNESS_EVAL_PROMPT_CONCISE_SYSTEM, GROUNDEDNESS_EVAL_PROMPT_CONCISE_USER

class Groundedness(LlmEvaluator):
    _failure_threshold: float

    def __init__(
            self,
            failure_threshold = 0.85,
            **kwargs
        ):
        self._failure_threshold = failure_threshold
        super().__init__(
            system_message_template=GROUNDEDNESS_EVAL_PROMPT_CONCISE_SYSTEM,
            user_message_template=GROUNDEDNESS_EVAL_PROMPT_CONCISE_USER,
            **kwargs
        )

    @property
    def name(self) -> str:
        return "Groundedness"

    @property
    def display_name(self) -> str:
        return "Groundedness"
    
    @property
    def default_model(self) -> str:
        return "gpt-3.5-turbo"

    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.GROUNDEDNESS.value]

    @property
    def required_args(self) -> List[str]:
        return ["context", "response"]

    @property
    def examples(self):
        return []
    
    def reason(self, unsupported_sentences: List[str]) -> str:
        if (len(unsupported_sentences) > 0):
            unsupported_sentences_str = "\n- ".join(unsupported_sentences)
            return f"The following sentences don't have any supporting evidence:\n- {unsupported_sentences_str}"
        else:
            return f"All sentences have sufficient supporting evidence. The answer is grounded."

    def _evaluate(self, **kwargs) -> EvalResult:
        """
        Run the LLM evaluator.
        """
        start_time = time.perf_counter()
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
            result = chat_completion_response_json["result"] # Pass / Fail - we ask the LLM to come up with a verdict but not using this for now.
            explanation = chat_completion_response_json["explanation"]
            groundedness_score_with_reason = GroundednessScore.compute(explanation)
            groundedness_score = groundedness_score_with_reason[0]
            unsupported_sentences = groundedness_score_with_reason[1]
            failure = groundedness_score < self._failure_threshold
            metrics.append(EvalResultMetric(id=MetricType.GROUNDEDNESS.value, value=groundedness_score))
            reason = self.reason(unsupported_sentences)

        except Exception as e:
            logger.error(f"Error occurred during eval: {e}")
            raise e

        end_time = time.perf_counter()
        eval_runtime_ms = int((end_time - start_time) * 1000)
        llm_eval_result = EvalResult(
            name=self.name,
            display_name=self.display_name,
            data=kwargs,
            failure=failure,
            reason=reason,
            runtime=eval_runtime_ms,
            model=self._model,
            metrics=metrics,
        )
        return {k: v for k, v in llm_eval_result.items() if v is not None}
    