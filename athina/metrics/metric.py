from abc import ABC, abstractmethod

class Metric(ABC):
    """
    Abstract base class for evaluation metrics.
    """

    @abstractmethod
    def compute(self):
        """
        Computes the metric.
        """
        pass