from abc import ABC, abstractmethod
import time
from typing import List, Optional
from athina.interfaces.result import LlmEvalResult
from athina.interfaces.model import Model
from athina.llms.openai_service import OpenAiService
from athina.helpers.logger import logger
from athina.helpers.json import JsonHelper
from athina.loaders import DataPoint
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
    def name(self):
        """A unique name identifier for the evaluator."""
        pass

    @property
    @abstractmethod
    def display_name(self):
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
        return self.EXAMPLES

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

    def run(self, **kwargs) -> LlmEvalResult:
        start_time = time.time()

        # Log usage to Athina for analytics
        AthinaApiService.log_usage(eval_name=self.name(), datapoints=1)

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
            failure = result == "Fail"
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

    def _run_batch_generator(self, data: List[DataPoint]):
        """
        Generator function for running a batch of evaluations.
        Iterates over a dataset, and runs the evaluator on each entry.
        """
        for entry in data:
            try:
                yield self.run(**entry)
            except Exception as e:
                logger.error(f"Error evaluating entry {entry}: {e}")
                yield None

    def run_batch(self, data: List[DataPoint]) -> List[LlmEvalResult]:
        """
        Runs the evaluator on a batch of data.
        """
        self._validate_batch_args(data)
        eval_results = list(self._run_batch_generator(data))

        # Log evaluation results to Athina
        if AthinaApiKey.is_set():
            AthinaApiService.log_eval_results(eval_results)

        return eval_results
