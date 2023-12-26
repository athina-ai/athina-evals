from typing import List, Optional
from athina.interfaces.result import EvalResult, EvalPerformanceReport
from athina.services.athina_api_service import AthinaApiService
from athina.helpers.logger import logger

class EvalPerformanceCalculator:
    """
    Class for calculating the performance metrics of the evaluator.
    """

    @staticmethod
    def calculate_eval_performance_metrics(
        eval_results: List[EvalResult],
        labels: List[bool],
        eval_request_id: Optional[str] = None,
        should_print: bool = False,
        should_log: bool = True,
    ) -> EvalPerformanceReport:
        """
        Calculates and logs the performance metrics for the evaluator.
        """

        if (labels is None) or (len(labels) != len(eval_results)):
            raise ValueError(
                "Labels must be provided and must be the same length as eval_results."
            )

        # Extract predictions from eval_results
        predictions = [result["failure"] for result in eval_results]

        # Initialize counters
        TP, FP, TN, FN = 0, 0, 0, 0

        # Count TP, FP, TN, FN
        for pred, label in zip(predictions, labels):
            if pred == 1 and label == 1:
                TP += 1
            elif pred == 1 and label == 0:
                FP += 1
            elif pred == 0 and label == 0:
                TN += 1
            elif pred == 0 and label == 1:
                FN += 1

        # Calculate metrics
        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall)
            else 0
        )
        runtime = int(sum([result["runtime"] for result in eval_results]))
        dataset_size = len(eval_results)

        if should_print:
            logger.info(f"Dataset Size: {dataset_size}")
            logger.info(f"Runtime: {runtime}")
            logger.info(f"True Positives: {TP}")
            logger.info(f"False Positives: {FP}")
            logger.info(f"True Negatives: {TN}")
            logger.info(f"False Negatives: {FN}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")
            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"F1 Score: {f1_score}")

        report = EvalPerformanceReport(
            true_positives=TP,
            false_positives=FP,
            true_negatives=TN,
            false_negatives=FN,
            precision=precision,
            recall=recall,
            accuracy=accuracy,
            f1_score=f1_score,
            dataset_size=dataset_size,
            runtime=runtime,
        )

        if eval_request_id is not None and should_log:
            AthinaApiService().log_eval_performance_report(eval_request_id, report)

        return report
