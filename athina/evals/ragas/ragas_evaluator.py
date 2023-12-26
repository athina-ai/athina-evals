
from typing import Optional
from athina.interfaces.model import Model
import time
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from athina.interfaces.result import EvalResult, EvalResultMetric, BatchRunResult
from athina.interfaces.model import Model
from athina.helpers.logger import logger
from athina.helpers.athina_logging_helper import AthinaLoggingHelper
from athina.interfaces.data import DataPoint
from athina.services.athina_api_service import AthinaApiService
from ..base_evaluator import BaseEvaluator
from datasets import Dataset
from ragas.llms import LangchainLLM
from langchain.chat_models import ChatOpenAI
from ragas import evaluate


class RagasEvaluator(BaseEvaluator):
    _model: str

    def __init__(
        self,
        model: Optional[str] = None,
    ):
        if model is None:
            self._model = self.default_model
        elif not Model.is_supported(model):
            raise ValueError(f"Unsupported model: {model}")
        else:
            self._model = model

    def _validate_args(self, **kwargs) -> None:
        for arg in self.required_args:
            if arg not in kwargs:
                raise ValueError(f"Missing required argument: {arg}")

    def _evaluate(self, **kwargs) -> EvalResult:
        """
        Run the Ragas evaluator.
        """
        start_time = time.time()

        # Validate that correct args were passed
        # self._validate_args(**kwargs)
        try:
            from ragas.metrics import context_relevancy

             # Set LLM model
            chat_model = ChatOpenAI(model_name=self._model)
            context_relevancy.llm = LangchainLLM(llm=chat_model)

            # Create a dataset from the test case
            data = {
                "contexts": [["France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower. The country is also renowned for its wines and sophisticated cuisine. Lascaux’s ancient cave drawings, Lyon’s Roman theater and the vast Palace of Versailles attest to its rich history"]],
                "question": ["What is the capital of France?"],
            }
            dataset = Dataset.from_dict(data)

            # Evaluate the dataset using Ragas
            scores = evaluate(dataset, metrics=[context_relevancy])

            # Ragas only does dataset-level comparisons
            context_relevancy_score = scores["context_relevancy"]
            print(context_relevancy_score)
        except Exception as e:
            logger.error(f"Error occurred during eval: {e}")
            raise e

        end_time = time.time()
        eval_runtime_ms = int((end_time - start_time) * 1000)
        llm_eval_result = EvalResult(
            name=self.name,
            display_name=self.display_name,
            data=kwargs,
            reason=explanation,
            runtime=eval_runtime_ms,
            model=self._model,
            metric=metric,
        )
        return {k: v for k, v in llm_eval_result.items() if v is not None}

    def run(self, **kwargs) -> BatchRunResult:
        """
        Run the Ragas evaluator, and log results to Athina.
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