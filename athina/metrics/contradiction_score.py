from .metric import Metric


class ContradictionScore(Metric):
    """
    Metric to evaluate the degree of contradiction between the answers obtained from
    a summary and the original document. It captures the percentage of questions that
    received contradictory answers between the summary and the document, with neither
    being 'Unknown'. A high score suggests the summary might be contradicting the
    original document's content.

    Attributes:
        answers_src (dict): Answers derived from the original document.
        answers_sum (dict): Answers derived from the summary.
        n_questions (int): Number of questions posed.
    """

    @staticmethod
    def _compute_metric(answers_src, answers_sum, questions):
        """
        Compute the number of contradictions between answers derived from the document
        and the summary.

        Args:
            answers_src (dict): Answers based on the original document.
            answers_sum (dict): Answers based on the summary.

        Returns:
            int: Number of contradictions.
        """
        answers_src_ls = list(answers_src.values())
        answers_sum_ls = list(answers_sum.values())

        n_contradiction = 0
        cont_questions = []

        for idx, (ans_src, ans_sum) in enumerate(zip(answers_src_ls, answers_sum_ls)):
            if (
                ans_src.strip().lower() in ["yes", "no"]
                and ans_src.strip().lower() != ans_sum.strip().lower()
            ):
                n_contradiction += 1
                cont_question = questions[idx]
                cont_questions.append(f"{cont_question}")

        return n_contradiction, cont_questions

    @staticmethod
    def compute(answers_src, answers_sum, questions, n_questions):
        """
        Compute the contradiction score by normalizing the number of contradictions by
        the total number of questions.

        Args:
            answers_src (dict): Answers based on the original document.
            answers_sum (dict): Answers based on the summary.
            n_questions (int): Total number of questions.

        Returns:
            float: Contradiction score.
        """
        n_contradiction, cont_questions = ContradictionScore._compute_metric(
            answers_src, answers_sum, questions
        )
        explanation = cont_questions
        cont_score = n_contradiction / n_questions
        return (cont_score, explanation)
