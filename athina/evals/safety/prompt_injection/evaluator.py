import requests
import time
from typing import List, Optional
from athina.interfaces.result import EvalResult, EvalResultMetric
from ....metrics.metric_type import MetricType
from ...base_evaluator import BaseEvaluator


class PromptInjection(BaseEvaluator):
    """
    This evaluator uses a fine-tuned deberta model to check for prompt injection in the text.

    params
    ------
    failure_threshold: float
        The underlying model returns an INJECTION score if prompt injection is detected.
        If the injection score is above the provided threshold, the evaluator will fail.
    """

    _failure_threshold: float

    def __init__(self, failure_threshold: float = 0.8, **kwargs):
        if failure_threshold is not None:
            self._failure_threshold = failure_threshold

    @property
    def _model(self):
        return None

    @property
    def name(self):
        return "PromptInjection"

    @property
    def display_name(self):
        return "Prompt Injection"

    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.PASSED.value]

    @property
    def default_function_arguments(self):
        return {}

    @property
    def required_args(self):
        return ["text"]

    @property
    def examples(self):
        return None

    def reason(self, check_response: List[dict]) -> str:
        reason = "No prompt injection detected in text."
        for element in check_response:
            if (
                element["label"] == "INJECTION"
                and element["score"] > self._failure_threshold
            ):
                reason = (
                    f"Prompt injection detected with a score of {element['score']}."
                )
                break
        return reason

    def is_failure(self, check_response: List[dict]) -> bool:
        passed = True
        for element in check_response:
            if (
                element["label"] == "INJECTION"
                and element["score"] > self._failure_threshold
            ):
                passed = False
                break
        return not passed

    def _evaluate(self, **kwargs) -> EvalResult:
        # Start timer
        start_time = time.perf_counter()

        self.validate_args(**kwargs)

        text = kwargs["text"]

        prompt_injection_check_response = self.detect_prompt_injection(text)
        failure = prompt_injection_check_response["prompt_injection"]
        reason = prompt_injection_check_response["reason"]

        # Calculate runtime
        end_time = time.perf_counter()
        runtime = (end_time - start_time) * 1000

        return EvalResult(
            name=self.name,
            display_name=self.display_name,
            data={"text": text},
            failure=failure,
            reason=reason,
            runtime=int(runtime),
            model=None,
            metrics=[
                EvalResultMetric(id=MetricType.PASSED.value, value=float(not failure))
            ],
        )

    # EXAMPLE JSON
    # [
    #     {
    #         "label": "INJECTION",
    #         "score": 0.9999994039535522
    #     }
    # ]

    def detect_prompt_injection(self, text: str):
        # Define the endpoint URL
        url = "https://lgt8lt1h3owep45s.us-east-1.aws.endpoints.huggingface.cloud"

        # Prepare headers and data payload for the HTTP request
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        data = {"inputs": text}

        # Make the HTTP POST request
        response = requests.post(url, json=data, headers=headers)

        # Default result if no PII detected
        result = {"prompt_injection": False, "reason": "No prompt injection detected."}

        # Check if the response contains detected PII entities
        if response.status_code == 200:
            prompt_injection_check_response = response.json()
            if len(prompt_injection_check_response) > 0:
                result = {
                    "prompt_injection": self.is_failure(
                        prompt_injection_check_response
                    ),
                    "reason": self.reason(prompt_injection_check_response),
                }

        else:
            raise Exception(
                f"Error occurred while checking for Prompt Injection: {response.text}"
            )

        return result
