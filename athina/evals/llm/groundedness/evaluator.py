import time
from typing import List, Tuple, Optional

from athina.interfaces.result import (
    EvalResult,
    EvalResultMetric,
    DatapointFieldAnnotation,
)
from athina.metrics.groundedness import GroundednessScore
from athina.helpers.logger import logger
from ....metrics.metric_type import MetricType
from ..llm_evaluator import LlmEvaluator
from .prompt import (
    GROUNDEDNESS_EVAL_PROMPT_CONCISE_SYSTEM,
    GROUNDEDNESS_EVAL_PROMPT_CONCISE_USER,
)


class Groundedness(LlmEvaluator):
    _failure_threshold: Optional[float] = None

    def __init__(self, failure_threshold: Optional[float] = None, **kwargs):
        super().__init__(
            system_message_template=GROUNDEDNESS_EVAL_PROMPT_CONCISE_SYSTEM,
            user_message_template=GROUNDEDNESS_EVAL_PROMPT_CONCISE_USER,
            **kwargs,
        )
        if failure_threshold is not None:
            self._failure_threshold = failure_threshold

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

    def is_failure(self, score) -> Optional[bool]:
        return (
            bool(score < self._failure_threshold)
            if self._failure_threshold is not None
            else None
        )

    def reason(self, unsupported_sentences: List[str]) -> str:
        if len(unsupported_sentences) > 0:
            unsupported_sentences_str = "\n- ".join(unsupported_sentences)
            return f"The following sentences don't have sufficient supporting evidence in the context:\n- {unsupported_sentences_str}"
        else:
            return f"All sentences have sufficient supporting evidence in the context. The answer is grounded."

    def datapoint_field_annotations(
        self,
        supported_sentences_with_evidence: List[Tuple[str, List[str]]],
        unsupported_sentences: List[str],
    ) -> List[DatapointFieldAnnotation]:
        datapoint_field_annotations = []
        for sentence, evidence in supported_sentences_with_evidence:
            evidences_str = "\n- ".join(evidence)
            datapoint_field_annotations.append(
                DatapointFieldAnnotation(
                    field_name="response",
                    text=sentence,
                    annotation_type="pass",
                    annotation_note=f"Supporting evidence:\n- {evidences_str}",
                )
            )
        for sentence in unsupported_sentences:
            datapoint_field_annotations.append(
                DatapointFieldAnnotation(
                    field_name="response",
                    text=sentence,
                    annotation_type="fail",
                    annotation_note="Not supported by any evidence in the context.",
                )
            )

        return datapoint_field_annotations

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
            result = chat_completion_response_json[
                "result"
            ]  # Pass / Fail - we ask the LLM to come up with a verdict but not using this for now.
            explanation = chat_completion_response_json["explanation"]
            groundedness_score_with_reason = GroundednessScore.compute(explanation)
            groundedness_score = groundedness_score_with_reason[0]
            unsupported_sentences = groundedness_score_with_reason[1]
            supported_sentences_with_evidence = groundedness_score_with_reason[
                2
            ]  # list of (sentices, evidence) pairs
            failure = self.is_failure(groundedness_score)
            metrics.append(
                EvalResultMetric(
                    id=MetricType.GROUNDEDNESS.value, value=groundedness_score
                )
            )
            reason = self.reason(unsupported_sentences)
            datapoint_field_annotations = self.datapoint_field_annotations(
                supported_sentences_with_evidence, unsupported_sentences
            )

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
            datapoint_field_annotations=datapoint_field_annotations,
        )
        return {k: v for k, v in llm_eval_result.items() if v is not None}

    def _user_message(
        self,
        context: List[str],
        response: str,
        **kwargs,
    ) -> str:
        """
        Generates data for evaluation.

        :param context: list of strings of retrieved context
        :param response: llm response
        :return: A dictionary with formatted data for evaluation
        """
        joined_context = "\n".join(context)
        return self._user_message_template.format(
            context=joined_context,
            response=response,
            examples=self._examples_str(),
        )
