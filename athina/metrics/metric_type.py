from enum import Enum
from .agreement_score import AgreementScore

class MetricType(Enum):
    AGREEMENT_SCORE = 'agreement_score'

    def get_class(self):
        if self == MetricType.AGREEMENT_SCORE:
            return AgreementScore
        else:
            raise NotImplementedError(f"Metric type {self} not implemented.")
