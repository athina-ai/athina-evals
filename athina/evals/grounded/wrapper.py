
from athina.evals.grounded.grounded_evaluator import GroundedEvaluator
from athina.evals.grounded.similarity import Comparator

class AnswerSimilarity(GroundedEvaluator):

    @property
    def required_args(self):
        return ["response", "expected_response"]


    def __init__(self, comparator: Comparator):
        """
        Initialize the grounded evaluator with a particular comparator.

        Args:
            comparator (Comparator): Concrete comparator to be used for comparison.
        
        Example:
            >>> AnswerSimilarity(comparator=CosineSimilarity())

        """
        super().__init__(
            comparator=comparator,
        )

class ContextSimilarity(GroundedEvaluator):
    
    @property
    def required_args(self):
        return ["response", "context"]


    def __init__(self, comparator: Comparator):
        """
        Initialize the grounded evaluator with a particular comparator.

        Args:
            comparator (Comparator): Concrete comparator to be used for comparison.
        
        Example:
            >>> ContextSimilarity(comparator=NormalisedLevenshteinSimilarity())

        """
        super().__init__(
            comparator=comparator,
        )