from enum import Enum
from .agreement_score import AgreementScore
from .hallucination_score import HallucinationScore
from .contradiction_score import ContradictionScore
from .ragas_metric import RagasMetric
from .passed import Passed
from .metric import Metric

class MetricType(Enum):
    AGREEMENT_SCORE = "agreement_score"
    HALLUCINATION_SCORE = "hallucination_score"
    CONTRADICTION_SCORE = "contradiction_score"
    RAGAS_CONTEXT_RELEVANCY = "ragas_context_relevancy"
    RAGAS_CONTEXT_PRECISION = "ragas_context_precision"
    RAGAS_ANSWER_RELEVANCY = "ragas_answer_relevancy"
    RAGAS_FAITHFULNESS = "ragas_faithfulness"
    RAGAS_HARMFULNESS = "ragas_harmfulness"
    RAGAS_MALICIOUSNESS = "ragas_maliciousness"
    RAGAS_COHERENCE = "ragas_coherence"
    RAGAS_CONTEXT_RECALL = "ragas_context_recall"
    RAGAS_ANSWER_SEMANTIC_SIMILARITY = "ragas_answer_semantic_similarity"
    RAGAS_ANSWER_CORRECTNESS = "ragas_answer_correctness"
    PASSED = 'passed'

    @staticmethod
    def get_class(metric_type):
        """
        Returns the class of the metric type.
        """
        if metric_type == MetricType.AGREEMENT_SCORE.value:
            return AgreementScore
        elif metric_type == MetricType.HALLUCINATION_SCORE.value:
            return HallucinationScore
        elif metric_type == MetricType.CONTRADICTION_SCORE.value:
            return ContradictionScore
        elif (metric_type == MetricType.RAGAS_CONTEXT_RELEVANCY.value or
                metric_type == MetricType.RAGAS_CONTEXT_PRECISION.value or
                metric_type == MetricType.RAGAS_ANSWER_RELEVANCY.value or 
                metric_type == MetricType.RAGAS_FAITHFULNESS.value or 
                metric_type == MetricType.RAGAS_CONTEXT_RECALL.value or 
                metric_type == MetricType.RAGAS_ANSWER_SEMANTIC_SIMILARITY.value or
                metric_type == MetricType.RAGAS_ANSWER_CORRECTNESS.value or
                metric_type == MetricType.RAGAS_HARMFULNESS or
                metric_type == MetricType.RAGAS_COHERENCE):
            return RagasMetric
        elif metric_type == MetricType.PASSED.value:
            return Passed
        else:
            raise NotImplementedError(f"Metric type {metric_type} not implemented.")
