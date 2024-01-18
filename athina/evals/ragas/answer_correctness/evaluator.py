from typing import List
from ..ragas_evaluator import RagasEvaluator
from athina.evals.eval_type import RagasEvalTypeId
from athina.metrics.metric_type import MetricType
from ragas.metrics import answer_correctness

"""
RAGAS Answer Correctness Docs: https://docs.ragas.io/en/latest/concepts/metrics/answer_correctness.html
RAGAS Answer Correctness Github: https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/_answer_correctness.py
"""
class RagasAnswerCorrectness(RagasEvaluator):
    """
    This evaluator involves gauging the accuracy of the generated llm response when compared to the ground truth
    """
    @property
    def name(self):
        return RagasEvalTypeId.RAGAS_ANSWER_CORRECTNESS.value

    @property
    def display_name(self):
        return "Answer Correctness"

    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.RAGAS_ANSWER_CORRECTNESS.value]
    
    @property
    def ragas_metric(self):
        return answer_correctness
    
    @property
    def ragas_metric_name(self):
        return "answer_correctness"

    @property
    def default_model(self):
        return "gpt-4-1106-preview"

    @property
    def required_args(self):
        return ["query", "response", "expected_response"]
    
    @property
    def examples(self):
        return None
    
    @property
    def grade_reason(self) -> str:
        return "Answer correctness encompasses two critical aspects: semantic similarity between the generated answer and the ground truth, as well as factual similarity. These aspects are combined using a weighted scheme to formulate the answer correctness score"
    
    def generate_data_to_evaluate(self, query, response, expected_response, **kwargs) -> dict:
        """
        Generates data for evaluation.

        :param query: user query
        :param response: llm response
        :param expected_response: expected output
        :return: A dictionary with formatted data for evaluation
        """
        data = {
            "question": [query],
            "answer": [response],
            "ground_truths": [[expected_response]]
        }
        return data
