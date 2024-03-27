from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from athina.helpers.logger import logger
from athina.helpers.athina_logging_helper import AthinaLoggingHelper
from athina.interfaces.athina import AthinaExperiment
from athina.interfaces.data import DataPoint
from athina.interfaces.result import BatchRunResult, EvalResult, GuardResult
from athina.services.athina_api_service import AthinaApiService
import traceback


class BaseEvaluator(ABC):
    _experiment: Optional[AthinaExperiment] = None

    # Abstract properties
    @property
    @abstractmethod
    def name(self) -> str:
        """A unique name identifier for the evaluator."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """A display name for the evaluator."""
        pass

    @property
    @abstractmethod
    def metric_ids(self) -> List[str]:
        """The metric computed by the evaluator."""
        pass

    @property
    @abstractmethod
    def required_args(self) -> List[str]:
        """A list of required arguments for the evaluator."""
        pass

    @property
    @abstractmethod
    def examples(self):
        """A list of examples for the evaluator."""
        pass

    @abstractmethod
    def is_failure(self, *args) -> Optional[bool]:
        """A method to determine if the evaluation failed."""
        pass

    @abstractmethod
    def _evaluate(self, **kwargs) -> EvalResult:
        """The method that performs the evaluation."""
        pass

    # Common methods
    def _examples_str(self) -> str:
        return "" if self.examples is None else "\n".join(map(str, self.examples))

    def configure_experiment(self, experiment: AthinaExperiment):
        """Configured metadata parameters to log an experiment to Athina"""
        self._experiment = experiment
        return self

    def validate_args(self, **kwargs) -> None:
        """
        Validates that all required arguments are present and not None.
        """
        for arg in self.required_args:
            if arg not in kwargs:
                raise ValueError(f"Missing required argument: {arg}")
            elif kwargs[arg] is None:
                raise ValueError(f"{arg} cannot be None")

    def _validate_batch_args(self, data: List[DataPoint]) -> bool:
        """
        Validates that each entry in the batch has all the required arguments,
        and none of the arguments is None.
        """
        for i, entry in enumerate(data):
            for arg in self.required_args:
                if arg not in entry:
                    raise ValueError(
                        f"Data at index {i} is missing required argument: {arg}"
                    )
                elif entry[arg] is None:
                    raise ValueError(
                        f"Data at index {i} has required argument {arg} set to None"
                    )
        return True

    def _log_evaluation_request(self, data) -> Optional[str]:
        """
        Logs usage to Athina for analytics and creates an evaluation request.
        """
        eval_request_id = None
        try:
            eval_request_id = AthinaLoggingHelper.create_eval_request(
                eval_name=self.name, request_data={"data": data}, request_type="batch"
            )
            self._log_experiment(eval_request_id)
        except Exception as e:
            pass
        return eval_request_id

    def _log_experiment(self, eval_request_id: Optional[str]):
        """
        Logs experiment to Athina if there is an ongoing experiment.
        """
        if self._experiment and eval_request_id:
            AthinaLoggingHelper.log_experiment(
                eval_request_id=eval_request_id,
                experiment=self._experiment,
            )

    def _log_evaluation_results(
        self, eval_request_id: Optional[str], eval_results: List[EvalResult]
    ):
        """
        Logs the evaluation results to Athina if the eval_request_id is available.
        """
        if eval_request_id:
            try:
                AthinaLoggingHelper.log_eval_results(
                    eval_request_id=eval_request_id,
                    eval_results=eval_results,
                )
            except Exception as e:
                pass

    def run(self, **kwargs) -> BatchRunResult:
        """
        Run the LLM evaluator, and log results to Athina.
        """
        AthinaApiService.log_usage(eval_name=self.name, run_type="batch")
        eval_request_id = self._log_evaluation_request(kwargs)
        eval_result = self._evaluate(**kwargs)
        self._log_evaluation_results(
            eval_request_id=eval_request_id, eval_results=[eval_result]
        )

        return BatchRunResult(
            eval_request_id=eval_request_id,
            eval_results=[eval_result],
        )

    def guard(self, **kwargs):
        """
        Guard
        """
        eval_result = self._evaluate(**kwargs)
        passed = not eval_result["failure"]
        reason = eval_result["reason"]
        runtime = eval_result["runtime"]
        return GuardResult(passed=passed, reason=reason, runtime=runtime)

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
                    logger.error(f"Error running batch async {entry}: {e}")
                    traceback.print_exc()
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
                traceback.print_exc()
                yield None

    def run_batch(
        self, data: List[DataPoint], max_parallel_evals: int = 5
    ) -> BatchRunResult:
        """
        Runs the evaluator on a batch of data.
        """
        # Log usage to Athina for analytics
        AthinaApiService.log_usage(eval_name=self.name, run_type="batch")
        eval_request_id = self._log_evaluation_request(data)

        # Run the evaluations
        if max_parallel_evals > 1:
            eval_results = self._run_batch_generator_async(data, max_parallel_evals)
        else:
            eval_results = list(self._run_batch_generator(data))

        # Log evaluation results to Athina
        self._log_evaluation_results(eval_request_id, eval_results)

        return BatchRunResult(
            eval_request_id=eval_request_id,
            eval_results=eval_results,
        )
