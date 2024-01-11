from .metric import Metric


class RagasAnswerRelevancy(Metric):
    """
    Float metric indicating the ragas answer relevancy metric score.
    """

    @staticmethod
    def compute(value: float):
        """
        Computes the result.

        Returns:
            float: Returns the metric
        """
        return value
