from enum import Enum
from .agreement_score import AgreementScore
from .hallucination_score import HallucinationScore
from .contradiction_score import ContradictionScore

class MetricType(Enum):
    AGREEMENT_SCORE = 'agreement_score'
    HALLUCINATION_SCORE = 'hallucination_score'
    CONTRADICTION_SCORE = 'contradiction_score'
    CONTEXT_SUFFICIENCY = 'context_sufficiency'
    ANSWER_COMPLETENESS = 'answer_completeness'
    RESPONSE_FAITHFULNESS = 'response_faithfulness'
    RAGAS_CONTEXT_RELEVANCY = 'ragas_context_relevancy'

    def get_class(self):
        if self == MetricType.AGREEMENT_SCORE:
            return AgreementScore
        if self == MetricType.HALLUCINATION_SCORE:
            return HallucinationScore
        if self == MetricType.CONTRADICTION_SCORE:
            return ContradictionScore
        else:
            raise NotImplementedError(f"Metric type {self} not implemented.")
