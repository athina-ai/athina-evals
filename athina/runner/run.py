from typing import List, TypedDict, Optional
from athina.datasets.dataset import Dataset
from athina.helpers.athina_logging_helper import AthinaLoggingHelper
from athina.evals.llm.llm_evaluator import LlmEvaluator
from athina.evals.base_evaluator import BaseEvaluator
from athina.helpers.dataset_helper import generate_unique_dataset_name, generate_eval_display_name
from athina.interfaces.result import EvalResult, BatchRunResult
from athina.interfaces.data import DataPoint
from athina.interfaces.athina import AthinaExperiment
from athina.services.athina_api_service import AthinaApiService
import pandas as pd
import json
import hashlib


class DataPointWithEvalResults(TypedDict):
    """A data point with its evaluation results."""

    data_point: DataPoint
    eval_results: List[EvalResult]


class LlmEvaluatorDescription(TypedDict):
    """A description of an LLM evaluator."""

    name: str
    display_name: str


class LlmBatchEvalResult(TypedDict):
    """Result of running a batch of LLM evaluations."""

    results: List[EvalResult]
    total_runtime: float
    passed_evals: int
    failed_evals: int
    total_evals: int
    total_datapoints: int


class EvalRunner:
    @staticmethod
    def eval_results_link(eval_request_id: str):
        return f"https://app.athina.ai/develop/request/{eval_request_id}"

    @staticmethod
    def flatten_eval_results(batch_eval_results) -> List:
        # Flatten the list of lists into a single list of evaluation results
        flattened_results = [
            item
            for sublist in batch_eval_results
            for item in (sublist if sublist is not None else [None])
        ]
        return flattened_results

    @staticmethod
    def _create_eval_request(eval_suite_name: str, data) -> Optional[str]:
        try:
            eval_request_id = AthinaLoggingHelper.create_eval_request(
                eval_name=eval_suite_name,
                request_data={"data": data},
                request_type="suite",
            )
            return eval_request_id
        except Exception as e:
            return None

    @staticmethod
    def _log_experiment(experiment, eval_request_id: Optional[str]):
        try:
            if experiment is not None and eval_request_id is not None:
                AthinaLoggingHelper.log_experiment(
                    eval_request_id=eval_request_id,
                    experiment=experiment,
                )
        except Exception as e:
            pass

    @staticmethod
    def _log_evaluation_results(
        eval_results: List[Optional[EvalResult]], eval_request_id: Optional[str]
    ):
        if eval_request_id:
            try:
                AthinaLoggingHelper.log_eval_results(
                    eval_request_id=eval_request_id,
                    eval_results=eval_results,
                )
            except Exception as e:
                pass

    @staticmethod
    def to_df(batch_eval_results):
        # Initialize a dictionary to hold the aggregated data
        aggregated_data = {}

        flattened_results = EvalRunner.flatten_eval_results(
            batch_eval_results=batch_eval_results
        )
        # Process each evaluation result
        for eval_result in flattened_results:
            if eval_result is not None:
                # Serialize and hash the datapoint dictionary to create a unique identifier
                datapoint_hash = hashlib.md5(
                    json.dumps(eval_result["data"], sort_keys=True).encode()
                ).hexdigest()

                # Initialize the datapoint in the aggregated data if not already present
                if datapoint_hash not in aggregated_data:
                    aggregated_data[datapoint_hash] = eval_result[
                        "data"
                    ]  # Include datapoint details

                # Update the aggregated data with metrics from this evaluation
                for metric in eval_result["metrics"]:
                    metric_name = metric["id"]
                    metric_value = metric["value"]
                    aggregated_data[datapoint_hash][
                        eval_result["display_name"] + " " + metric_name
                    ] = metric_value

        # Convert the aggregated data into a DataFrame
        df = pd.DataFrame(list(aggregated_data.values()))

        return df

    @staticmethod
    def _log_eval_results_with_config(
        eval_results: List[dict], eval: BaseEvaluator, dataset_id: str
    ):
        try:
            eval_config = eval.to_config()
            llm_engine = getattr(eval, "_model", None)
            AthinaLoggingHelper.log_eval_results_with_config(
                eval_results_with_config={
                    "eval_results": eval_results,
                    "development_eval_config": {
                        "eval_type_id": eval.name,
                        "eval_display_name": generate_eval_display_name(eval.display_name),
                        "eval_config": eval_config,
                        "llm_engine": llm_engine,
                    },
                },
                dataset_id=dataset_id,
            )
        except Exception as e:
            print(
                f"An error occurred while posting eval results",
                str(e),
            )
            raise

    @staticmethod
    def _log_dataset_to_athina(data: List[DataPoint]) -> Optional[str]:
        """
        Logs the dataset to Athina
        """
        try:
            dataset = Dataset.create(name=generate_unique_dataset_name(), rows=data)
            return dataset
        except Exception as e:
            print(f"Error logging dataset to Athina: {e}")
            return None

    @staticmethod
    def _fetch_dataset_rows(dataset_id: str, number_of_rows: Optional[int] = None) -> List[any]:
        """
        Fetch the dataset rows from Athina
        """
        try:
            rows = Dataset.fetch_dataset_rows(dataset_id=dataset_id, number_of_rows=number_of_rows)
            return rows
        except Exception as e:
            print(f"Error fetching dataset rows: {e}")
            return None

    @staticmethod
    def run_suite(
        evals: List[BaseEvaluator],
        data: List[DataPoint] = None,
        max_parallel_evals: int = 5,
        dataset_id: Optional[str] = None,
        number_of_rows: Optional[int] = None,
    ) -> List[LlmBatchEvalResult]:
        """
        Run a suite of LLM evaluations against a dataset.

        Args:
            evals: A list of LlmEvaluator objects.
            data: A list of data points.

        Returns:
            A list of LlmBatchEvalResult objects.
        """
        eval_suite_name = "llm_eval_suite" + "_" + ",".join(eval.name for eval in evals)
        AthinaApiService.log_usage(eval_name=eval_suite_name, run_type="suite")

        if data:
            # Log Dataset to Athina
            dataset = EvalRunner._log_dataset_to_athina(data)
            dataset_id = dataset.id
        elif dataset_id is not None:
            dataset = EvalRunner._fetch_dataset_rows(dataset_id, number_of_rows)
            data = dataset
        else:
            raise Exception("No data or dataset_id provided.")  

        batch_results = []
        for eval in evals:
            # Run the evaluations
            if max_parallel_evals > 1:
                eval_results = eval._run_batch_generator_async(data, max_parallel_evals)
            else:
                eval_results = list(eval._run_batch_generator(data))

            if dataset:
                EvalRunner._log_eval_results_with_config(
                    eval_results=eval_results, eval=eval, dataset_id=dataset_id
                )
            batch_results.append(eval_results)

        if dataset:
            print(f"You can view your dataset at: {Dataset.dataset_link(dataset_id)}")

        return EvalRunner.to_df(batch_results)
