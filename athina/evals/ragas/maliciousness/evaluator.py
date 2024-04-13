from athina.interfaces.model import Model
from ..ragas_evaluator import RagasEvaluator
from athina.evals.eval_type import RagasEvalTypeId
from athina.metrics.metric_type import MetricType
from ragas.metrics.critique import maliciousness
from typing import List, Optional

"""
RAGAS Maliciousness Docs: https://docs.ragas.io/en/latest/concepts/metrics/critique.html
RAGAS Maliciousness Github: https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/critique.py
"""
class RagasMaliciousness(RagasEvaluator):
    """
    This measures if the generated response intends to harm, deceive, or exploit users
    """
    @property
    def name(self):
        return RagasEvalTypeId.RAGAS_MALICIOUSNESS.value

    @property
    def display_name(self):
        return "Ragas Maliciousness"

    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.RAGAS_MALICIOUSNESS.value]
    
    @property
    def ragas_metric(self):
        return maliciousness
    
    @property
    def ragas_metric_name(self):
        return "maliciousness"

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
        return "This is calculated by how much potential generated response has to harm, deceive, or exploit users"

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
            "answer": [response],
        }
        return data
