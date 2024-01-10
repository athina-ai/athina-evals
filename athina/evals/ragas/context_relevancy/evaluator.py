from typing import List
from ..ragas_evaluator import RagasEvaluator
from athina.evals.eval_type import RagasEvalTypeId
from athina.metrics.metric_type import MetricType
from ragas.metrics import context_relevancy


class RagasContextRelevancy(RagasEvaluator):
    """
    This evaluator calculates the relevancy of the context with respect to the user query.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        return "gpt-4-1106-preview"

    @property
    def required_args(self):
        return ["query", "context"]

    @property
    def examples(self):
        return None
    
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
