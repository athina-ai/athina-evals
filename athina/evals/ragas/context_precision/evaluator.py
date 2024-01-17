from athina.interfaces.model import Model
from ..ragas_evaluator import RagasEvaluator
from athina.evals.eval_type import RagasEvalTypeId
from athina.metrics.metric_type import MetricType
from ragas.metrics import context_precision
from typing import List

"""
RAGAS Context Precision Docs: https://docs.ragas.io/en/latest/concepts/metrics/context_precision.html
RAGAS Context Precision Github: https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/_context_precision.py
"""
class RagasContextPrecision(RagasEvaluator):
    """
    This evaluator calculates the precision of the context with respect to the expected response.
    Context Precision is a metric that evaluates whether all of the ground-truth relevant items present in the context are ranked higher or not. 
    Ideally all the relevant chunks must appear at the top ranks.
    """
    @property
    def name(self):
        return RagasEvalTypeId.RAGAS_CONTEXT_PRECISION.value

    @property
    def display_name(self):
        return "Context Precision"

    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.RAGAS_CONTEXT_PRECISION.value]
    
    @property
    def ragas_metric(self):
        return context_precision
    
    @property
    def ragas_metric_name(self):
        return "context_precision"

    @property
    def default_model(self):
        return Model.GPT4_1106_PREVIEW.value

    @property
    def required_args(self):
        return ["query", "contexts", "expected_response"]

    @property
    def examples(self):
        return None
    
    @property
    def grade_reason(self) -> str:
        return "This metric evaluates whether all of the ground-truth relevant items present in the context are ranked higher or not. Ideally all the relevant chunks must appear at the top ranks."
    
    def generate_data_to_evaluate(self, contexts, query, expected_response, **kwargs) -> dict:
        """
        Generates data for evaluation.

        :param contexts: list of strings of retrieved context
        :param query: user query
        :param expected_response: expected output
        :return: A dictionary with formatted data for evaluation
        """
        data = {
            "contexts": [contexts],
            "question": [query],
            "ground_truths": [[expected_response]]
        }
        return data
