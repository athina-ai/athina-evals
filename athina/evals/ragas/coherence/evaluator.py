from athina.interfaces.model import Model
from ..ragas_evaluator import RagasEvaluator
from athina.evals.eval_type import RagasEvalTypeId
from athina.metrics.metric_type import MetricType
from ragas.metrics.critique import coherence
from typing import List, Optional

"""
RAGAS Coherence Docs: https://docs.ragas.io/en/latest/concepts/metrics/critique.html
RAGAS Coherence Github: https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/critique.py
"""


class RagasCoherence(RagasEvaluator):
    """
    This evaluates if the generated llm response presents ideas, information, or arguments in a logical and organized manner
    """

    @property
    def name(self):
        return RagasEvalTypeId.RAGAS_COHERENCE.value

    @property
    def display_name(self):
        return "Ragas Coherence"

    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.RAGAS_COHERENCE.value]

    @property
    def ragas_metric(self):
        return coherence

    @property
    def ragas_metric_name(self):
        return "coherence"

    @property
    def default_model(self):
        return Model.GPT35_TURBO.value

    @property
    def required_args(self):
        return ["response"]

    @property
    def examples(self):
        return None

    @property
    def grade_reason(self) -> str:
        return "This is calculated by how coherent is the generated llm response and how able it is able to present ideas, information, or arguments in a logical and organized manner"

    def is_failure(self, score) -> Optional[bool]:
        return (
            bool(score < self._failure_threshold)
            if self._failure_threshold is not None
            else None
        )

    def generate_data_to_evaluate(self, response, **kwargs) -> dict:
        """
        Generates data for evaluation.
        :param response: llm response
        :return: A dictionary with formatted data for evaluation.
        """
        data = {"contexts": [[""]], "question": [""], "answer": [response]}
        return data
