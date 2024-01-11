from .metric import Metric


class RagasContextRelevancy(Metric):
    """
    Float metric indicating the ragas context relevancy metric score.
    """

    @staticmethod
    def compute(value: float):
        """
        Computes the result.

        Returns:
            float: Returns the metric
        """
        return value
