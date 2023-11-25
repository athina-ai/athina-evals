from typing import List, TypedDict
from athina.evals.llm.llm_evaluator import LlmEvaluator, LlmEvalResult
from athina.loaders.loader import DataPoint


class DataPointWithEvalResults(TypedDict):
    """A data point with its evaluation results."""

    data_point: DataPoint
    eval_results: List[LlmEvalResult]


class LlmEvaluatorDescription(TypedDict):
    """A description of an LLM evaluator."""

    name: str
    display_name: str


class LlmBatchEvalResult(TypedDict):
    """Result of running a batch of LLM evaluations."""

    results: List[DataPointWithEvalResults]
    evals: List[LlmEvaluatorDescription]
    total_runtime: float
    passed_evals: int
    failed_evals: int
    total_evals: int
    total_datapoints: int


class EvalRunner:
    @staticmethod
    def batch_eval_result(
        datapoints_with_eval_results: List[DataPointWithEvalResults],
        eval_descriptions: List[LlmEvaluatorDescription],
    ) -> LlmBatchEvalResult:
        """
        Calculate metrics for a batch of LLM evaluations.

        Args:
            datapoints_with_eval_results: A list of DataPointWithEvalResults objects.

        Returns:
            A LlmBatchEvalResult object.
        """
        total_runtime = 0
        passed_evals = 0
        failed_evals = 0
        total_evals = 0

        # Iterate through each DataPointWithEvalResults
        for datapoint in datapoints_with_eval_results:
            for eval_result in datapoint.get("eval_results", []):
                total_evals += 1
                total_runtime += eval_result.get("runtime", 0)

                # Counting passed and failed evaluations
                if eval_result.get("failure"):
                    failed_evals += 1
                else:
                    passed_evals += 1

        total_datapoints = len(datapoints_with_eval_results)

        return LlmBatchEvalResult(
            evals=eval_descriptions,
            total_runtime=total_runtime,
            passed_evals=passed_evals,
            failed_evals=failed_evals,
            total_evals=total_evals,
            total_datapoints=total_datapoints,
            results=datapoints_with_eval_results,
        )

    @staticmethod
    def run_batch(
        evals: List[LlmEvaluator],
        dataset: List[DataPoint],
    ) -> List[LlmBatchEvalResult]:
        """
        Run a suite of LLM evaluations against a dataset.

        Args:
            evals: A list of LlmEvaluator objects.
            data: A list of data points.

        Returns:
            A list of LlmBatchEvalResult objects.
        """
        datapoints_with_eval_results = []
        for datapoint in dataset:
            eval_results = []
            for evaluator in evals:
                try:
                    eval_result = evaluator.run(**datapoint)
                    eval_result.pop("data", None)
                    eval_results.append(eval_result)
                except Exception as e:
                    print(f"Error evaluating entry {datapoint}: {e}")
            datapoints_with_eval_results.append(
                {
                    "data_point": datapoint,
                    "eval_results": eval_results,
                }
            )
        eval_descriptions = list(
            map(
                lambda x: {
                    "name": x.NAME,
                    "display_name": x.DISPLAY_NAME,
                },
                evals,
            )
        )
        return EvalRunner.batch_eval_result(
            datapoints_with_eval_results=datapoints_with_eval_results,
            eval_descriptions=eval_descriptions,
        )
