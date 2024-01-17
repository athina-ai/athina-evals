from .metric import Metric


class RagasMetric(Metric):
    """
    Float ragas metric
    """

    @staticmethod
    def compute(value: float):
        """
        Computes the result.

        Returns:
            float: Returns the metric
        """
        return value
