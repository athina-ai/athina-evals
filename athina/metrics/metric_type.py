from enum import Enum
from .agreement_score import AgreementScore
from .hallucination_score import HallucinationScore
from .contradiction_score import ContradictionScore
from .ragas_context_relevancy import RagasContextRelevancy
from .ragas_answer_relevancy import RagasAnswerRelevancy
from .passed import Passed

class MetricType(Enum):
    AGREEMENT_SCORE = "agreement_score"
    HALLUCINATION_SCORE = "hallucination_score"
    CONTRADICTION_SCORE = "contradiction_score"
    RAGAS_CONTEXT_RELEVANCY = "ragas_context_relevancy"
    RAGAS_ANSWER_RELEVANCY = "ragas_answer_relevancy"
    PASSED = 'passed'

    def get_class(self):
        if self == MetricType.AGREEMENT_SCORE:
            return AgreementScore
        elif self == MetricType.HALLUCINATION_SCORE:
            return HallucinationScore
        elif self == MetricType.CONTRADICTION_SCORE:
            return ContradictionScore
        elif self == MetricType.RAGAS_CONTEXT_RELEVANCY:
            return RagasContextRelevancy
        elif self == MetricType.RAGAS_ANSWER_RELEVANCY:
            return RagasAnswerRelevancy
        elif self == MetricType.PASSED:
            return Passed
        else:
            raise NotImplementedError(f"Metric type {self} not implemented.")
