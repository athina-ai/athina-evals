from athina.interfaces.model import Model
from ..ragas_evaluator import RagasEvaluator
from athina.evals.eval_type import RagasEvalTypeId
from athina.metrics.metric_type import MetricType
from ragas.metrics.critique import harmfulness
from typing import List, Optional

"""
RAGAS Harmfulness Docs: https://docs.ragas.io/en/latest/concepts/metrics/critique.html
RAGAS Harmfulness Github: https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/critique.py
"""
class RagasHarmfulness(RagasEvaluator):
    """
    This measures if the generated response has the potential to cause harm to individuals, groups, or society at large
    """
    @property
    def name(self):
        return RagasEvalTypeId.RAGAS_HARMFULNESS.value

    @property
    def display_name(self):
        return "Ragas Harmfulness"

    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.RAGAS_HARMFULNESS.value]
    
    @property
    def ragas_metric(self):
        return harmfulness
    
    @property
    def ragas_metric_name(self):
        return "harmfulness"

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
        return "This is calculated by how much potential generated response has to cause harm to individuals, groups, or society at large"

    def is_failure(self, score) -> Optional[bool]:
        return bool(score > self._failure_threshold) if self._failure_threshold is not None else None
        
    def generate_data_to_evaluate(self, context, query, response, **kwargs) -> dict:
        """
        Generates data for evaluation.

        :param context: list of strings of retrieved context
        :param query: user query
        :param response: llm response
        :return: A dictionary with formatted data for evaluation.
        """
        data = {
            "contexts": [context],
            "question": [query],
            "answer": [response]
        }
        return data
