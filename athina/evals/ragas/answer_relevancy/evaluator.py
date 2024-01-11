from typing import List
from ..ragas_evaluator import RagasEvaluator
from athina.evals.eval_type import RagasEvalTypeId
from athina.metrics.metric_type import MetricType
from ragas.metrics import answer_relevancy


class RagasAnswerRelevancy(RagasEvaluator):
    """
    This evaluator focuses on assessing how pertinent the generated response is to the given prompt. 
    A lower score is assigned to responses that are incomplete or contain redundant information.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return RagasEvalTypeId.RAGAS_ANSWER_RELEVANCY.value

    @property
    def display_name(self):
        return "Answer Relevancy"

    @property
    def metric_ids(self) -> str:
        return [MetricType.RAGAS_CONTEXT_RELEVANCY.value]
    
    @property
    def ragas_metric(self):
        return answer_relevancy
    
    @property
    def ragas_metric_name(self):
        return "answer_relevancy"

    @property
    def default_model(self):
        return "gpt-4-1106-preview"

    @property
    def required_args(self):
        return ["query", "context", "response"]

    @property
    def examples(self):
        return None
    
    def generate_data_to_evaluate(self, query, context, response,  **kwargs) -> dict:
        """
        Generates data for evaluation.

        :param context: context.
        :param query: query.
        :param response: llm response
        :return: A dictionary with formatted data for evaluation.
        """
        data = {
            "contexts": [[str(context)]],
            "question": [query],
            "answer": [response]
        }
        return data