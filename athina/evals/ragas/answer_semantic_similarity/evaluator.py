from athina.interfaces.model import Model
from ..ragas_evaluator import RagasEvaluator
from athina.evals.eval_type import RagasEvalTypeId
from athina.metrics.metric_type import MetricType
from ragas.metrics import answer_similarity
from typing import List, Optional

"""
RAGAS Answer Semantic Similarity Docs: https://docs.ragas.io/en/latest/concepts/metrics/semantic_similarity.html
RAGAS Answer Semantid Similarity Github: https://github.com/explodinggradients/ragas/blob/main/src/ragas/metrics/_answer_similarity.py
"""
class RagasAnswerSemanticSimilarity(RagasEvaluator):
    """
    This evaluator measures the semantic resemblance between the generated llm response and the ground truth. 
    """
    @property
    def name(self):
        return RagasEvalTypeId.RAGAS_ANSWER_SEMANTIC_SIMILARITY.value

    @property
    def display_name(self):
        return "Ragas Answer Semantic Similarity"

    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.RAGAS_ANSWER_SEMANTIC_SIMILARITY.value]
    
    @property
    def ragas_metric(self):
        return answer_similarity
    
    @property
    def ragas_metric_name(self):
        return "answer_similarity"

    @property
    def default_model(self):
        return Model.GPT35_TURBO.value

    @property
    def required_args(self):
        return ["response", "expected_response"]

    @property
    def examples(self):
        return None
    
    @property
    def grade_reason(self) -> str:
        return "Answer Semantic Similarity pertains to the assessment of the semantic resemblance between the generated response and the ground truth. This evaluation is based on the ground truth and the response, with values falling within the range of 0 to 1. A higher score signifies a better alignment between the generated response and the ground truth"
    
    def is_failure(self, score) -> Optional[bool]:
        return bool(score < self._failure_threshold) if self._failure_threshold is not None else None
        
    def generate_data_to_evaluate(self, response, expected_response, **kwargs) -> dict:
        """
        Generates data for evaluation.

        :param response: llm response
        :param expected_response: expected output
        :return: A dictionary with formatted data for evaluation
        """
        data = {
            "answer": [response],
            "ground_truths": [[expected_response]]
        }
        return data
