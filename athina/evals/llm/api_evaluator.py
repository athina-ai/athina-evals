import time
from typing import List
from athina.helpers import logger
from athina.interfaces.result import LlmEvalResult
from athina.loaders import DataPoint
from athina.services.athina_api_service import AthinaApiService
from .llm_evaluator import LlmEvaluator


class ApiEvaluator(LlmEvaluator):
    """
    This class is meant to be extended by evaluators that use the Athina API to trigger evals.

    Overrides the run and run_batch methods of LlmEvaluator.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, **kwargs) -> LlmEvalResult:
        """Override the run method of LlmEvaluator to trigger eval via Athina API"""
        start_time = time.time()

        # Log usage to Athina for analytics
        AthinaApiService.log_usage(eval_name=self.name())

        # Validate that correct args were passed
        self._validate_args(**kwargs)

        # Construct Prompt
        messages = self._prompt_messages(**kwargs)

        # Run the eval
        try:
            # Send API request to trigger eval
            eval_request_id = AthinaApiService.trigger_eval(
                eval_name=self.name(), eval_params=[kwargs]
            )

            # Poll for eval result
            result = None
            while result is None or result.get("status") != "completed":
                result = AthinaApiService.get_eval_result(
                    eval_request_id=eval_request_id
                )
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error occurred during eval: {e}")
            raise e

        end_time = time.time()
        eval_runtime_ms = int((end_time - start_time) * 1000)

        # TODO: Safely extract results and transform API result into LlmEvalResult
        llm_eval_result = result.get("eval_results")[0]

        return llm_eval_result

    def run_batch(self, data: List[DataPoint]) -> List[LlmEvalResult]:
        """Override the run_batch method of LlmEvaluator to trigger eval via Athina API"""
        start_time = time.time()

        # Validate that correct args were passed
        self._validate_batch_args(data)

        # Log usage to Athina for analytics
        AthinaApiService.log_usage(eval_name=self.name(), datapoints=len(data))

        # Run the eval
        try:
            # Send API request to trigger eval
            eval_request_id = AthinaApiService.trigger_eval(
                eval_name=self.name(), eval_params=data
            )

            # Poll for eval result
            result = None
            while result is None or result.get("status") != "completed":
                result = AthinaApiService.get_eval_result(
                    eval_request_id=eval_request_id
                )
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error occurred during eval: {e}")
            raise e

        end_time = time.time()
        eval_runtime_ms = int((end_time - start_time) * 1000)

        # TODO: Safely extract results and transform API result into LlmEvalResult
        llm_eval_results = result.get("eval_results")

        return llm_eval_results
