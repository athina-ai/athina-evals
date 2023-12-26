from ..ragas_evaluator import RagasEvaluator
from athina.evals.eval_type import RagasEvalTypeId
from athina.metrics.metric_type import MetricType


class ContextRelevancy(RagasEvaluator):
    """
    This evaluator calculates the relevancy of the context with respect to the user query.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return RagasEvalTypeId.CONTEXT_RELEVANCY.value

    @property
    def display_name(self):
        return "Context Relevancy"

    @property
    def metric_id(self) -> str:
        return MetricType.CONTEXT_RELEVANCY.value

    @property
    def default_model(self):
        return "gpt-4-1106-preview"

    @property
    def required_args(self):
        return ["query", "context"]

    @property
    def examples(self):
        return None
