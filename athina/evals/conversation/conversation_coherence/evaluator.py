import time
from typing import List, Optional, Dict

from athina.helpers.logger import logger
from athina.interfaces.model import Model
from athina.interfaces.result import EvalResult, EvalResultMetric
from athina.evals.llm.llm_evaluator import LlmEvaluator
from athina.evals.eval_type import ConversationEvalTypeId
from athina.metrics.metric_type import MetricType
from .prompt import SYSTEM_MESSAGE, USER_MESSAGE


class ConversationCoherence(LlmEvaluator):
    """
    This evaluator checks if the conversation was resolved or not.
    """

    _failure_threshold: Optional[float] = None

    def __init__(self, failure_threshold: Optional[float] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if failure_threshold is not None:
            self._failure_threshold = failure_threshold
        self._system_message_template = SYSTEM_MESSAGE
        self._user_message_template = USER_MESSAGE

    @property
    def name(self):
        return ConversationEvalTypeId.CONVERSATION_COHERENCE.value

    @property
    def display_name(self):
        return "Conversation Coherence"

    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.CONVERSATION_COHERENCE.value]

    @property
    def default_model(self):
        return Model.GPT35_TURBO.value

    @property
    def required_args(self):
        return [
            "messages"
        ]  # messages is an array of strings representing the conversation

    @property
    def examples(self):
        return []

    def _user_message(self, **kwargs) -> str:
        return self._user_message_template.format(**kwargs)

    def is_failure(self, score) -> Optional[bool]:
        return (
            bool(score < self._failure_threshold)
            if self._failure_threshold is not None
            else None
        )

    def score(self, details):
        """Calculate the percentage of coherent messages."""
        total_messages = len(details)
        coherent_messages = sum(detail["result"] == "coherent" for detail in details)
        if total_messages > 0:
            return coherent_messages / total_messages
        else:
            return 0

    def reason(self, details):
        """Construct a string listing all non-coherent messages."""
        non_coherent_messages = [
            detail["message"]
            for detail in details
            if detail["result"] == "not_coherent"
        ]
        if non_coherent_messages:
            return "The following messages were not coherent: " + ", ".join(
                non_coherent_messages
            )
        else:
            return "All messages were coherent."

    def _evaluate(self, messages: List[str]) -> EvalResult:
        """
        Run the LLM evaluator.
        """
        start_time = time.perf_counter()

        # Construct Prompt
        prompt_messages = self._prompt_messages(messages="\n".join(messages))

        # Run the LLM Completion
        chat_completion_response_json: dict = self.llm_service.json_completion(
            model=self._model,
            messages=prompt_messages,
            temperature=self.TEMPERATURE,
        )

        metrics = []
        try:
            messages_with_coherence_status = chat_completion_response_json["details"]

            score = self.score(messages_with_coherence_status)
            reason = self.reason(messages_with_coherence_status)

            metrics.append(
                EvalResultMetric(
                    id=MetricType.CONVERSATION_COHERENCE.value, value=score
                )
            )
            failure = self.is_failure(score=score)

        except Exception as e:
            logger.error(f"Error occurred during eval: {e}")
            raise e

        end_time = time.perf_counter()
        eval_runtime_ms = int((end_time - start_time) * 1000)
        llm_eval_result = EvalResult(
            name=self.name,
            display_name=self.display_name,
            data={"messages": messages},
            failure=failure,
            reason=reason,
            runtime=eval_runtime_ms,
            model=self._model,
            metrics=metrics,
            datapoint_field_annotations=None,
        )
        return {k: v for k, v in llm_eval_result.items() if v is not None}
