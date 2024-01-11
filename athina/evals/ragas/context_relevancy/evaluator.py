from athina.interfaces.model import Model
from ..ragas_evaluator import RagasEvaluator
from athina.evals.eval_type import RagasEvalTypeId
from athina.metrics.metric_type import MetricType
from ragas.metrics import context_relevancy

"""
RAGAS Context Relevancy Docs: https://docs.ragas.io/en/latest/concepts/metrics/context_relevancy.html
RAGAS Context Relevancy Github: https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/_context_relevancy.py
"""
class RagasContextRelevancy(RagasEvaluator):
    """
    This evaluator calculates the relevancy of the context with respect to the user query.
    """
    @property
    def name(self):
        return RagasEvalTypeId.RAGAS_CONTEXT_RELEVANCY.value

    @property
    def display_name(self):
        return "Context Relevancy"

    @property
    def metric_ids(self) -> str:
        return [MetricType.RAGAS_CONTEXT_RELEVANCY.value]
    
    @property
    def ragas_metric(self):
        return context_relevancy
    
    @property
    def ragas_metric_name(self):
        return "context_relevancy"

    @property
    def default_model(self):
        return Model.GPT4_1106_PREVIEW.value

    @property
    def required_args(self):
        return ["query", "context"]

    @property
    def examples(self):
        return None
    
    @property
    def grade_reason(self):
        return "This metric is calulated by dividing the number of sentences in context that are relevant for answering the given query by the total number of sentences in the retrieved context"
    
    def generate_data_to_evaluate(self, context, query, **kwargs) -> dict:
        """
        Generates data for evaluation.

        :param context: context.
        :param query: query.
        :return: A dictionary with formatted data for evaluation.
        """
        data = {
            "contexts": [[str(context)]],
            "question": [query]
        }
        return data
