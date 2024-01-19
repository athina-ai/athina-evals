from enum import Enum

from athina.metrics.groundedness import GroundednessScore
from .agreement_score import AgreementScore
from .hallucination_score import HallucinationScore
from .contradiction_score import ContradictionScore
from .ragas_context_relevancy import RagasContextRelevancy
from .ragas_answer_relevancy import RagasAnswerRelevancy
from .passed import Passed
from .metric import Metric

class MetricType(Enum):
    AGREEMENT_SCORE = "agreement_score"
    HALLUCINATION_SCORE = "hallucination_score"
    CONTRADICTION_SCORE = "contradiction_score"
    RAGAS_CONTEXT_RELEVANCY = "ragas_context_relevancy"
    RAGAS_ANSWER_RELEVANCY = "ragas_answer_relevancy"
    GROUNDEDNESS = "groundedness"
    PASSED = 'passed'

    @staticmethod
    def get_class(metric_type):
        """
        Returns the class of the metric type.
        """
        if metric_type == MetricType.AGREEMENT_SCORE.value:
            return AgreementScore
        if metric_type == MetricType.GROUNDEDNESS.value:
            return GroundednessScore
        elif metric_type == MetricType.HALLUCINATION_SCORE.value:
            return HallucinationScore
        elif metric_type == MetricType.CONTRADICTION_SCORE.value:
            return ContradictionScore
        elif metric_type == MetricType.RAGAS_CONTEXT_RELEVANCY.value:
            return RagasContextRelevancy
        elif metric_type == MetricType.RAGAS_ANSWER_RELEVANCY.value:
            return RagasAnswerRelevancy
        elif metric_type == MetricType.PASSED.value:
            return Passed
        else:
            raise NotImplementedError(f"Metric type {metric_type} not implemented.")
