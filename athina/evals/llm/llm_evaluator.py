from abc import ABC, abstractmethod
import time
from typing import List, Optional
from athina.interfaces.athina import (
    AthinaEvalResult,
    AthinaEvalRunResult,
    AthinaJobType,
    AthinaInterfaceHelper,
    AthinaEvalRequestCreateRequest,
    AthinaEvalRequestSource,
)
from athina.interfaces.result import LlmEvalResult, EvalPerformanceMetrics
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

    TEMPERATURE = 0.0

    RETURN_FORMAT_INSTRUCTIONS = """
    Provide a brief explanation of why the response does or does not pass the grading criteria, labeled as 'explanation', leading up to a verdict (Pass/Fail) labeled as 'result'.

    You MUST return a JSON object in the following format: { "result": 'result', "explanation": 'explanation' }.
    """

    SYSTEM_MESSAGE_TEMPLATE = f""" 
    ### INSTRUCTIONS ###
    You are an expert at evaluating chatbot responses, according to some grading criteria.

    If it passes the grading criteria, then your result is Pass, otherwise it is Fail.
    """

    USER_MESSAGE_TEMPLATE = """
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

    def _system_message(self, **kwargs) -> str:
        return self.SYSTEM_MESSAGE_TEMPLATE + self.RETURN_FORMAT_INSTRUCTIONS

    def _user_message(self, **kwargs) -> str:
        return self.USER_MESSAGE_TEMPLATE.format(
            examples=self._examples_str(),
            grading_criteria=self.grading_criteria,
            **kwargs,
        )

    def _prompt_messages(self, **kwargs) -> List[dict]:
        return [
            {
                "role": "system",
                "content": self._system_message(**kwargs),
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
            result = chat_completion_response_json["result"]
            explanation = chat_completion_response_json["explanation"]
            failure = bool(result == "Fail")
        except Exception as e:
            logger.error(f"Error occurred during eval: {e}")
            raise e

        end_time = time.time()
        eval_runtime_ms = int((end_time - start_time) * 1000)

        return LlmEvalResult(
            name=self.name(),
            data=kwargs,
            failure=failure,
            reason=explanation,
            runtime=eval_runtime_ms,
            model=self.model,
        )

    def set_experiment_configuration(self, **kwargs) -> None:
        """Configured metadata parameters to log an experiment to Athina"""
        self._experiment_settings(
            experiment_name=self.name(),
            eval_type=AthinaJobType.LLM_EVAL.value,
            eval_request_source=AthinaEvalRequestSource.DEV_SDK.value,
            eval_request_data=kwargs,
        )

    def run(self, **kwargs) -> LlmEvalResult:
        """
        Run the LLM evaluator, and log results to Athina.
        """
        # Log usage to Athina for analytics
        AthinaApiService.log_usage(eval_name=self.name())

        eval_request_id = AthinaLoggingHelper.create_eval_request(
            eval_name=self.name(), request_data=kwargs, request_type="single"
        )

        eval_result = self._evaluate(**kwargs)

        AthinaLoggingHelper.log_eval_results(
            eval_request_id=eval_request_id,
            eval_results=[eval_result],
            data=[kwargs],
        )

        return eval_result

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
        self, data: List[DataPoint], labels: Optional[List[bool]] = None
    ) -> List[LlmEvalResult]:
        """
        Runs the evaluator on a batch of data.
        """

        # Create eval request
        eval_request_id = AthinaLoggingHelper.create_eval_request(
            eval_name=self.name(), request_data={"data": data}, request_type="batch"
        )

        self._validate_batch_args(data)
        eval_results = list(self._run_batch_generator(data))

        if (labels is not None) and (len(labels) == len(eval_results)):
            self.calculate_eval_performance_metrics(
                eval_results=eval_results, labels=labels
            )

        # Log evaluation results to Athina
        AthinaLoggingHelper.log_eval_results(
            eval_request_id=eval_request_id,
            eval_results=eval_results,
            data=data,
        )

        return eval_results

    def calculate_eval_performance_metrics(
        self,
        eval_results: List[LlmEvalResult],
        labels: List[bool],
    ) -> EvalPerformanceMetrics:
        """
        Calculates the performance metrics for the evaluator.
        """

        # Extract predictions from eval_results
        predictions = [result["failure"] for result in eval_results]

        # Initialize counters
        TP, FP, TN, FN = 0, 0, 0, 0

        # Count TP, FP, TN, FN
        for pred, label in zip(predictions, labels):
            if pred == 1 and label == 1:
                TP += 1
            elif pred == 1 and label == 0:
                FP += 1
            elif pred == 0 and label == 0:
                TN += 1
            elif pred == 0 and label == 1:
                FN += 1

        # Calculate metrics
        precision = TP / (TP + FP) if (TP + FP) else 0
        recall = TP / (TP + FN) if (TP + FN) else 0
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall)
            else 0
        )

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Accuracy: {accuracy}")
        print(f"F1 Score: {f1_score}")

        return EvalPerformanceMetrics(
            precision=precision,
            recall=recall,
            accuracy=accuracy,
            f1_score=f1_score,
        )
