from typing import List, TypedDict
from athina.helpers.athina_logging_helper import AthinaLoggingHelper
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
    def run_suite(
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
        # Create eval request
        eval_suite_name = "llm_eval_suite"
        eval_request_id = AthinaLoggingHelper.create_eval_request(
            eval_name=eval_suite_name,
            request_data={"data": dataset},
            request_type="suite",
        )

        datapoints_with_eval_results = []
        for datapoint in dataset:
            eval_results = []
            for evaluator in evals:
                try:
                    eval_result = evaluator._evaluate(**datapoint)
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
                    "name": x.name(),
                    "display_name": x.display_name(),
                },
                evals,
            )
        )

        # Log evaluation results to Athina
        eval_suite_results = list(
            map(lambda x: x["eval_results"], datapoints_with_eval_results)
        )
        flattened_eval_results = [
            item for sublist in eval_suite_results for item in sublist
        ]

        dataset = list(map(lambda x: x["data_point"], datapoints_with_eval_results))

        AthinaLoggingHelper.log_eval_results(
            eval_request_id=eval_request_id,
            eval_results=flattened_eval_results,
            data=dataset,
        )

        return EvalRunner.batch_eval_result(
            datapoints_with_eval_results=datapoints_with_eval_results,
            eval_descriptions=eval_descriptions,
        )
