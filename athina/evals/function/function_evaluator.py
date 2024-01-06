
from typing import Optional

from athina.evals.eval_type import FunctionEvalTypeId
from athina.metrics.metric_type import MetricType
from athina.interfaces.model import Model
import time
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from athina.interfaces.result import EvalResult, EvalResultMetric, BatchRunResult
from athina.interfaces.model import Model
from athina.helpers.logger import logger
from athina.helpers.athina_logging_helper import AthinaLoggingHelper
from athina.interfaces.data import DataPoint
from athina.interfaces.athina import AthinaExperiment
from athina.services.athina_api_service import AthinaApiService
from ..base_evaluator import BaseEvaluator
from .functions import operations

class FunctionEvaluator(BaseEvaluator):

    _function_name: str
    _function_arguments: dict
    _experiment: Optional[AthinaExperiment] = None
    _model: str

    """
    This evaluator runs the requested Function on the given data.
    """

    @property
    def _model(self):
        return ""
    
    @property
    def name(self):
        return self._function_name

    @property
    def display_name(self):
        return "Function evaluator"

    @property
    def metric_id(self) -> str:
        return MetricType.PASSED.value

    @property
    def default_model(self):
        return None

    @property
    def default_function(self):
        return "contains_any"

    @property
    def default_function_arguments(self):
        return {}

    @property
    def required_args(self):
        return ["response"]

    @property
    def examples(self):
        return None

    def __init__(
        self,
        function_name: Optional[str] = None,
        function_arguments: Optional[dict] = None,
    ):
        if function_name is None:
            function_name = self.default_function
        if function_arguments is None:
            function_arguments = self.default_function_arguments
        if function_name not in operations.keys():
            raise ValueError(f"Unsupported function: {function_name}")
        else:
            self._function_name = function_name
            self._function_arguments = function_arguments

    def _validate_args(self, **kwargs) -> None:
        for arg in self.required_args:
            if arg not in kwargs:
                raise ValueError(f"Missing required argument: {arg}")

    def _evaluate(self, **kwargs) -> EvalResult:
        """
        Run the Function evaluator.
        """
        start_time = time.time()

        # Validate that correct args were passed
        self._validate_args(**kwargs)
        metrics = []
        try: 
            # Evaluate the dataset using Function
            operator = operations.get(self._function_name)
            response = operator(**kwargs, **self._function_arguments)
            metrics.append(EvalResultMetric(id=self.metric_id, value=float(not response["result"])))
            explanation = response['reason']

        except Exception as e:
            logger.error(f"Error occurred during eval: {e}")
            raise e

        end_time = time.time()
        eval_runtime_ms = int((end_time - start_time) * 1000)
        eval_result = EvalResult(
            name=self.name,
            display_name=self.display_name,
            data=kwargs,
            reason=explanation,
            runtime=eval_runtime_ms,
            model=self._model,
            metrics=metrics,
            failure=response["result"],
        )
        return {k: v for k, v in eval_result.items() if v is not None}

    def run(self, **kwargs) -> BatchRunResult:
        """
        Run the Function evaluator, and log results to Athina.
        """
        # Log usage to Athina for analytics
        AthinaApiService.log_usage(eval_name=self.name, run_type="single")

        # Create eval request
        eval_request_id = AthinaLoggingHelper.create_eval_request(
            eval_name=self.name, request_data=kwargs, request_type="single"
        )

        # Log experiment
        if self._experiment:
            AthinaLoggingHelper.log_experiment(
                eval_request_id=eval_request_id,
                experiment=self._experiment,
            )

        eval_result = self._evaluate(**kwargs)

        # Log evaluation results to Athina
        AthinaLoggingHelper.log_eval_results(
            eval_request_id=eval_request_id,
            eval_results=[eval_result],
        )

        return BatchRunResult(
            eval_request_id=eval_request_id,
            eval_results=[eval_result],
        )

    def _validate_batch_args(self, data: List[DataPoint]) -> bool:
        """
        Validates that each entry in the batch has all the required arguments.
        """
        for i, entry in enumerate(data):
            for arg in self.required_args:
                if arg not in entry:
                    raise ValueError(
                        f"Data at index {i} is missing required argument: {arg}"
                    )
        return True

    def _run_batch_generator_async(
        self, data: List[DataPoint], max_parallel_evals: int
    ):
        with ThreadPoolExecutor(max_workers=max_parallel_evals) as executor:
            # Submit all tasks to the executor and store them with their original index
            future_to_index = {
                executor.submit(self._evaluate, **entry): i
                for i, entry in enumerate(data)
            }

            # Create a list to store results in the original order
            results = [None] * len(data)

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    entry = data[index]
                    logger.error(f"Error evaluating entry {entry}: {e}")
                    results[index] = None

            return results

    def _run_batch_generator(self, data: List[DataPoint]):
        """
        Generator function for running a batch of evaluations.
        Iterates over a dataset, and runs the evaluator on each entry.
        """
        for entry in data:
            try:
                yield self._evaluate(**entry)
            except Exception as e:
                logger.error(f"Error evaluating entry {entry}: {e}")
                yield None

    def run_batch(
        self, data: List[DataPoint], max_parallel_evals: int = 1
    ) -> BatchRunResult:
        """
        Runs the evaluator on a batch of data.
        """

        # Create eval request
        eval_request_id = AthinaLoggingHelper.create_eval_request(
            eval_name=self.name, request_data={"data": data}, request_type="batch"
        )

        # Log usage to Athina for analytics
        AthinaApiService.log_usage(eval_name=self.name, run_type="batch")

        # Log experiment
        if self._experiment is not None:
            AthinaLoggingHelper.log_experiment(
                eval_request_id=eval_request_id,
                experiment=self._experiment,
            )

        # Validate the dataset against the required args
        self._validate_batch_args(data)

        # Run the evaluations
        if max_parallel_evals > 1:
            eval_results = self._run_batch_generator_async(data, max_parallel_evals)
        else:
            eval_results = list(self._run_batch_generator(data))

        # Log evaluation results to Athina
        AthinaLoggingHelper.log_eval_results(
            eval_request_id=eval_request_id,
            eval_results=eval_results,
        )

        return BatchRunResult(
            eval_request_id=eval_request_id,
            eval_results=eval_results,
        )