from enum import Enum
from .agreement_score import AgreementScore
from .hallucination_score import HallucinationScore
from .contradiction_score import ContradictionScore
from .passed import Passed

class MetricType(Enum):
    AGREEMENT_SCORE = 'agreement_score'
    HALLUCINATION_SCORE = 'hallucination_score'
    CONTRADICTION_SCORE = 'contradiction_score'
    PASSED = 'passed'

    def get_class(self):
        if self == MetricType.AGREEMENT_SCORE:
            return AgreementScore
        elif self == MetricType.HALLUCINATION_SCORE:
            return HallucinationScore
        elif self == MetricType.CONTRADICTION_SCORE:
            return ContradictionScore
        elif self == MetricType.PASSED:
            return Passed
        else:
            raise NotImplementedError(f"Metric type {self} not implemented.")
