from athina.interfaces.model import Model
from ..ragas_evaluator import RagasEvaluator
from athina.evals.eval_type import RagasEvalTypeId
from athina.metrics.metric_type import MetricType
from ragas.metrics.critique import conciseness
from typing import List

"""
RAGAS Conciseness Docs: https://docs.ragas.io/en/latest/concepts/metrics/critique.html
RAGAS Conciseness Github: https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/critique.py
"""
class RagasConciseness(RagasEvaluator):
    """
    This evaluates if the generated llm response conveys information or ideas clearly and efficiently, without unnecessary or redundant details
    """
    @property
    def name(self):
        return RagasEvalTypeId.RAGAS_CONCISENESS.value

    @property
    def display_name(self):
        return "Ragas Conciseness"

    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.RAGAS_CONCISENESS.value]
    
    @property
    def ragas_metric(self):
        return conciseness
    
    @property
    def ragas_metric_name(self):
        return "conciseness"

    @property
    def default_model(self):
        return Model.GPT4_1106_PREVIEW.value

    @property
    def required_args(self):
        return ["query", "contexts", "response", "expected_response"]

    @property
    def examples(self):
        return None
    
    @property
    def grade_reason(self) -> str:
        return "This is calculated by how efficiently generated llm response conveys information or ideas clearly and efficiently, without unnecessary or redundant details"

    def generate_data_to_evaluate(self, contexts, query, response, expected_response, **kwargs) -> dict:
        """
        Generates data for evaluation.

        :param context: list of strings of retrieved context
        :param query: user query
        :param response: llm response
        :param expected_response: expected output
        :return: A dictionary with formatted data for evaluation.
        """
        data = {
            "contexts": [contexts],
            "question": [query],
            "answer": [response],
            "ground_truths": [[expected_response]]
        }
        return data