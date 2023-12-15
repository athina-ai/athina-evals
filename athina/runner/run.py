from typing import List, TypedDict, Optional
from athina.helpers.athina_logging_helper import AthinaLoggingHelper
from athina.evals.llm.llm_evaluator import LlmEvaluator, LlmEvalResult
from athina.interfaces.result import LlmEvalResult, LlmEvalResultMetric, BatchRunResult
from athina.interfaces.data import DataPoint
from athina.interfaces.athina import AthinaExperiment
from athina.services.athina_api_service import AthinaApiService


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

    results: List[LlmEvalResult]
    total_runtime: float
    passed_evals: int
    failed_evals: int
    total_evals: int
    total_datapoints: int


class EvalRunner:
    @staticmethod
    def batch_eval_result(
        eval_results: List[LlmEvalResult],
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
        for eval_result in eval_results:
            total_evals += 1
            total_runtime += eval_result.get("runtime", 0)

            # Counting passed and failed evaluations
            if eval_result.get("failure"):
                failed_evals += 1
            else:
                passed_evals += 1

        total_datapoints = len(eval_result)

        return LlmBatchEvalResult(
            results=eval_result,
            total_runtime=total_runtime,
            passed_evals=passed_evals,
            failed_evals=failed_evals,
            total_evals=total_evals,
            total_datapoints=total_datapoints,
        )

    @staticmethod
    def run_suite(
        evals: List[LlmEvaluator],
        data: List[DataPoint],
        experiment: Optional[AthinaExperiment] = None,
        max_parallel_evals: int = 1,
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
        eval_suite_name = "llm_eval_suite" + "_" + ",".join(eval.name for eval in evals)
        eval_request_id = AthinaLoggingHelper.create_eval_request(
            eval_name=eval_suite_name,
            request_data={"data": data},
            request_type="suite",
        )

        # Log experiment
        if experiment is not None:
            AthinaLoggingHelper.log_experiment(
                eval_request_id=eval_request_id,
                experiment=experiment,
            )

        AthinaApiService.log_usage(eval_name=eval_suite_name, run_type="suite")

        batch_results = []
        for eval in evals:
            # Log usage to Athina for analytics

            # Validate the dataset against the required args
            eval._validate_batch_args(data)

            # Run the evaluations
            if max_parallel_evals > 1:
                eval_results = eval._run_batch_generator_async(data, max_parallel_evals)
            else:
                eval_results = list(eval._run_batch_generator(data))

            # Log evaluation results to Athina
            AthinaLoggingHelper.log_eval_results(
                eval_request_id=eval_request_id,
                eval_results=eval_results,
            )

            batch_results.append(BatchRunResult(
                eval_request_id=eval_request_id,
                eval_results=eval_results,
            ))

        return EvalRunner.batch_eval_result(eval_results=eval_results)
