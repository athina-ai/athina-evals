from athina.evals.grounded.grounded_evaluator import GroundedEvaluator
from athina.evals.grounded.similarity import Comparator


class AnswerSimilarity(GroundedEvaluator):

    @property
    def required_args(self):
        return ["response", "expected_response"]

    @property
    def name(self):
        return "AnswerSimilarity"

    def __init__(self, comparator: Comparator, failure_threshold: float = None):
        """
        Initialize the grounded evaluator with a particular comparator.

        Args:
            comparator (Comparator): Concrete comparator to be used for comparison.
            failure_threshold (float): Threshold for failure. If the similarity score is below this threshold it's marked as failed.
        Example:
            >>> AnswerSimilarity(comparator=CosineSimilarity())
            >>> AnswerSimilarity(comparator=CosineSimilarity(), failure_threshold=0.8)

        """
        super().__init__(comparator=comparator, failure_threshold=failure_threshold)


class ContextSimilarity(GroundedEvaluator):

    @property
    def required_args(self):
        return ["response", "context"]

    @property
    def name(self):
        return "ContextSimilarity"

    def __init__(self, comparator: Comparator, failure_threshold: float = None):
        """
        Initialize the grounded evaluator with a particular comparator.

        Args:
            comparator (Comparator): Concrete comparator to be used for comparison.
            failure_threshold (float): Threshold for failure. If the similarity score is below this threshold it's marked as failed.

        Example:
            >>> ContextSimilarity(comparator=NormalisedLevenshteinSimilarity())

        """
        super().__init__(comparator=comparator, failure_threshold=failure_threshold)
