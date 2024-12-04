from athina.interfaces.model import Model
from ..ragas_evaluator import RagasEvaluator
from athina.evals.eval_type import RagasEvalTypeId
from athina.metrics.metric_type import MetricType
from ragas.metrics import faithfulness
from typing import List, Optional

"""
RAGAS Faithfulness Docs: https://docs.ragas.io/en/latest/concepts/metrics/faithfulness.html
RAGAS Faithfulness Github: https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/_faithfulness.py
"""


class RagasFaithfulness(RagasEvaluator):
    """
    This measures the factual consistency of the generated response against the given context.
    """

    @property
    def name(self):
        return RagasEvalTypeId.RAGAS_FAITHFULNESS.value

    @property
    def display_name(self):
        return "Ragas Faithfulness"

    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.RAGAS_FAITHFULNESS.value]

    @property
    def ragas_metric(self):
        return faithfulness

    @property
    def ragas_metric_name(self):
        return "faithfulness"

    @property
    def default_model(self):
        return Model.GPT35_TURBO.value

    @property
    def required_args(self):
        return ["query", "context", "response"]

    @property
    def examples(self):
        return None

    @property
    def grade_reason(self) -> str:
        return "The generated answer is regarded as faithful if all the claims that are made in the answer can be inferred from the given context. To calculate this a set of claims from the generated answer is first identified. Then each one of these claims are cross checked with given context to determine if it can be inferred from given context or not"

    def is_failure(self, score) -> Optional[bool]:
        return (
            bool(score < self._failure_threshold)
            if self._failure_threshold is not None
            else None
        )

    def generate_data_to_evaluate(self, context, query, response, **kwargs) -> dict:
        """
        Generates data for evaluation.

        :param context: list of strings of retrieved context
        :param query: user query
        :param response: llm response
        :return: A dictionary with formatted data for evaluation.
        """
        data = {"contexts": [context], "question": [query], "answer": [response]}
        return data
