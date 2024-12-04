from athina.interfaces.model import Model
from ..ragas_evaluator import RagasEvaluator
from athina.evals.eval_type import RagasEvalTypeId
from athina.metrics.metric_type import MetricType
from ragas.metrics import context_recall
from typing import List, Optional

"""
RAGAS Context Recall Docs: https://docs.ragas.io/en/latest/concepts/metrics/context_recall.html
RAGAS Context Recall Github: https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/_context_recall.py
"""


class RagasContextRecall(RagasEvaluator):
    """
    This measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth.
    """

    @property
    def name(self):
        return RagasEvalTypeId.RAGAS_CONTEXT_RECALL.value

    @property
    def display_name(self):
        return "Ragas Context Recall"

    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.RAGAS_CONTEXT_RECALL.value]

    @property
    def ragas_metric(self):
        return context_recall

    @property
    def ragas_metric_name(self):
        return "context_recall"

    @property
    def default_model(self):
        return Model.GPT35_TURBO.value

    @property
    def required_args(self):
        return ["query", "context", "expected_response"]

    @property
    def examples(self):
        return None

    @property
    def grade_reason(self) -> str:
        return "Context Recall metric is calculated by dividing the number of sentences in the ground truth that can be attributed to retrieved context by the total number of sentences in the grouund truth"

    def is_failure(self, score) -> Optional[bool]:
        return (
            bool(score < self._failure_threshold)
            if self._failure_threshold is not None
            else None
        )

    def generate_data_to_evaluate(
        self, context, query, expected_response, **kwargs
    ) -> dict:
        """
        Generates data for evaluation.

        :param context: list of strings of retrieved context
        :param query: user query
        :param expected_response: expected output
        :return: A dictionary with formatted data for evaluation
        """
        data = {
            "contexts": [context],
            "question": [query],
            "ground_truths": [[expected_response]],
        }
        return data
