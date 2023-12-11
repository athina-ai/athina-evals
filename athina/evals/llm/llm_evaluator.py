from abc import ABC, abstractmethod
import time
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from athina.interfaces.result import LlmEvalResult, LlmEvalResultMetric, BatchRunResult
from athina.interfaces.openai import OpenAiPromptMessage
from athina.interfaces.athina import AthinaExperiment
from athina.interfaces.model import Model
from athina.llms.openai_service import OpenAiService
from athina.helpers.logger import logger
from athina.helpers.athina_logging_helper import AthinaLoggingHelper
from athina.helpers.json import JsonHelper
from athina.interfaces.data import DataPoint
from athina.keys import AthinaApiKey
from athina.errors.exceptions import NoAthinaApiKeyException
from athina.services.athina_api_service import AthinaApiService
from .example import FewShotExample


class LlmEvaluator(ABC):
    llm_service: OpenAiService
    grading_criteria: str
    _experiment: Optional[AthinaExperiment] = None
    _system_message_template: Optional[str] = None
    _user_message_template: Optional[str] = None

    TEMPERATURE = 0.0

    RETURN_FORMAT_INSTRUCTIONS = """
    You MUST return a JSON object with the following fields: 
    - result: Result must be either 'Pass' or 'Fail'.
    - explanation: An explanation of why the result is Pass or Fail.
    - score: (Optional) Use the scoring criteria specified.
    """

    DEFAULT_SYSTEM_MESSAGE_TEMPLATE = f""" 
    ### INSTRUCTIONS ###
    You are an expert at evaluating chatbot responses, according to some grading criteria.

    If it passes the grading criteria, then your result is Pass, otherwise it is Fail.
    
    """

    DEFAULT_USER_MESSAGE_TEMPLATE = """
    ### GRADING CRITERIA ###
    {grading_criteria}

    ### EXAMPLES ###
    {examples}

    ### RESPONSE TO EVALUATE ###
    {response}
    """

    EXAMPLES: FewShotExample = []

    def __init__(
        self,
        model: Optional[str] = None,
        grading_criteria: Optional[str] = None,
        system_message_template: Optional[str] = None,
        user_message_template: Optional[str] = None,
    ):
        if not AthinaApiKey.is_set():
            raise NoAthinaApiKeyException()

        self.llm_service = OpenAiService()
        self.grading_criteria = grading_criteria if grading_criteria else ""
        if model is None:
            self.model = self.default_model()
        elif not Model.is_supported(model):
            raise ValueError(f"Unsupported model: {model}")
        else:
            self.model = model

        # Initialize message templates
        if system_message_template is None:
            self._system_message_template = (
                self.DEFAULT_SYSTEM_MESSAGE_TEMPLATE + self.RETURN_FORMAT_INSTRUCTIONS
            )
        else:
            self._system_message_template = system_message_template

        if user_message_template is None:
            self._user_message_template = self.DEFAULT_USER_MESSAGE_TEMPLATE
        else:
            self._user_message_template = user_message_template

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
    def metric_id(self) -> str:
        """The metric computed by the evaluator."""
        pass

    @property
    @abstractmethod
    def required_args(self):
        """A list of required arguments for the evaluator."""
        pass

    @property
    @abstractmethod
    def default_model(self):
        """The default model for the evaluator."""
        pass

    @property
    @abstractmethod
    def examples(self):
        """A list of examples for the evaluator."""
        pass

    # Common methods
    def _examples_str(self) -> str:
        return "\n".join([str(example) for example in self.examples()])

    def _system_message(self) -> str:
        return self._system_message_template

    def _user_message(self, **kwargs) -> str:
        return self._user_message_template.format(
            examples=self._examples_str(),
            grading_criteria=self.grading_criteria,
            **kwargs,
        )

    def _prompt_messages(self, **kwargs) -> List[dict]:
        return [
            {
                "role": "system",
                "content": self._system_message(),
            },
            {
                "role": "user",
                "content": self._user_message(**kwargs),
            },
        ]

    def _validate_args(self, **kwargs) -> None:
        for arg in self.required_args():
            if arg not in kwargs:
                raise ValueError(f"Missing required argument: {arg}")

    def _evaluate(self, **kwargs) -> LlmEvalResult:
        """
        Run the LLM evaluator.
        """
        start_time = time.time()

        # Validate that correct args were passed
        self._validate_args(**kwargs)

        # Construct Prompt
        messages = self._prompt_messages(**kwargs)

        # Run the LLM Completion
        if Model.supports_json_mode(self.model):
            chat_completion_response = self.llm_service.json_completion(
                model=self.model,
                messages=messages,
                temperature=self.TEMPERATURE,
            )
        else:
            chat_completion_response = self.llm_service.chat_completion(
                model=self.model,
                messages=messages,
                temperature=self.TEMPERATURE,
            )

        # Extract JSON object from LLM response
        chat_completion_response_json = JsonHelper.extract_json_from_text(
            chat_completion_response
        )
        try:
            metric = None
            result = chat_completion_response_json["result"]
            explanation = chat_completion_response_json["explanation"]
            failure = bool(result == "Fail")
            if "score" in chat_completion_response_json:
                score = chat_completion_response_json["score"]
                metric = LlmEvalResultMetric(id=self.metric_id(), value=score)
            else:
                metric = LlmEvalResultMetric(id="failed", value=float(failure))

        except Exception as e:
            logger.error(f"Error occurred during eval: {e}")
            raise e

        end_time = time.time()
        eval_runtime_ms = int((end_time - start_time) * 1000)
        llm_eval_result = LlmEvalResult(
            name=self.name(),
            display_name=self.display_name(),
            data=kwargs,
            failure=failure,
            reason=explanation,
            runtime=eval_runtime_ms,
            model=self.model,
            metric=metric,
        )
        return {k: v for k, v in llm_eval_result.items() if v is not None}

    def configure_experiment(self, experiment: AthinaExperiment):
        """Configured metadata parameters to log an experiment to Athina"""
        self._experiment = experiment
        return self

    def run(self, **kwargs) -> BatchRunResult:
        """
        Run the LLM evaluator, and log results to Athina.
        """
        # Log usage to Athina for analytics
        AthinaApiService.log_usage(eval_name=self.name(), run_type="single")

        # Create eval request
        eval_request_id = AthinaLoggingHelper.create_eval_request(
            eval_name=self.name(), request_data=kwargs, request_type="single"
        )

        # Log experiment
        if self._experiment:
            AthinaApiService.log_experiment(
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
            for arg in self.required_args():
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
            eval_name=self.name(), request_data={"data": data}, request_type="batch"
        )

        # Log usage to Athina for analytics
        AthinaApiService.log_usage(eval_name=self.name(), run_type="batch")

        # Log experiment
        if self._experiment is not None:
            AthinaApiService.log_experiment(
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
