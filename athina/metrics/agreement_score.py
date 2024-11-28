from .metric import Metric


class AgreementScore(Metric):
    """
    Calculates agreement score between two sets of answers.

    AgreementScore computes the proportion of questions that received
    consistent answers between a source (e.g., document) and a summary.
    """

    @staticmethod
    def _compute_metric(answers_src, answers_sum, questions):
        """
        Computes the number of matches between the answers from source and summary.

        Args:
            answers_src (dict): Answers derived from the source.
            answers_sum (dict): Answers derived from the summary.

        Returns:
            int: Number of questions with consistent answers.
        """
        answers_src_ls = list(answers_src.values())
        answers_sum_ls = list(answers_sum.values())
        n_matches = 0
        agreed_questions = []
        for idx, (ans_src, ans_sum) in enumerate(zip(answers_src_ls, answers_sum_ls)):
            if ans_src.strip().lower() == ans_sum.strip().lower():
                n_matches += 1
                agreed_question = questions[idx]
                agreed_questions.append(f"{agreed_question}")
        return n_matches, agreed_questions

    @staticmethod
    def compute(answers_src, answers_sum, questions, n_questions):
        """
        Computes the agreement score.

        Args:
            answers_src (dict): Answers derived from the source.
            answers_sum (dict): Answers derived from the summary.
            n_questions (int): Total number of questions.

        Returns:
            float: Agreement score.
        """
        n_matches, agreed_questions = AgreementScore._compute_metric(
            answers_src, answers_sum, questions
        )
        explanation = agreed_questions
        agreement_score = n_matches / n_questions
        return agreement_score, explanation
