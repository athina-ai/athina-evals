from typing import List, Optional
from athina.interfaces.result import LlmEvalResult, EvalPerformanceReport
from athina.services.athina_api_service import AthinaApiService


class EvalPerformanceCalculator:
    """
    Class for calculating the performance metrics of the evaluator.
    """

    @staticmethod
    def calculate_eval_performance_metrics(
        eval_results: List[LlmEvalResult],
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
            print(f"Dataset Size: {dataset_size}")
            print(f"Runtime: {runtime}")
            print(f"True Positives: {TP}")
            print(f"False Positives: {FP}")
            print(f"True Negatives: {TN}")
            print(f"False Negatives: {FN}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"Accuracy: {accuracy}")
            print(f"F1 Score: {f1_score}")

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
