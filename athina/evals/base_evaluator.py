from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from athina.helpers.logger import logger
from athina.helpers.athina_logging_helper import AthinaLoggingHelper
from athina.interfaces.athina import AthinaExperiment

from athina.interfaces.data import DataPoint
from athina.interfaces.result import BatchRunResult
from athina.services.athina_api_service import AthinaApiService


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

    # Common methods
    def _examples_str(self) -> str:
        return "" if self.examples is None else "\n".join(map(str, self.examples))
    
    def configure_experiment(self, experiment: AthinaExperiment):
        """Configured metadata parameters to log an experiment to Athina"""
        self._experiment = experiment
        return self

    def validate_args(self, **kwargs) -> None:
        for arg in self.required_args:
            if arg not in kwargs:
                raise ValueError(f"Missing required argument: {arg}")

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

    def run(self, **kwargs) -> BatchRunResult:
        """
        Run the LLM evaluator, and log results to Athina.
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
        self, data: List[DataPoint], max_parallel_evals: int = 5
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