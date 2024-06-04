import requests
import time
from typing import List, Optional
from athina.interfaces.result import EvalResult, EvalResultMetric
from ....metrics.metric_type import MetricType
from ...base_evaluator import BaseEvaluator


class PiiDetection(BaseEvaluator):
    @property
    def _model(self):
        return None

    @property
    def name(self):
        return "PiiDetection"

    @property
    def display_name(self):
        return "PII Detection"

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

    def is_failure(self, detected_pii_response) -> Optional[bool]:
        return bool(detected_pii_response["pii_detected"])

    def _evaluate(self, **kwargs) -> EvalResult:
        # Start timer
        start_time = time.perf_counter()

        self.validate_args(**kwargs)

        text = kwargs["text"]
        detected_pii_response = self.detect_pii(text)
        failure = self.is_failure(detected_pii_response)
        reason = str(detected_pii_response["reason"])

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
    #         "entity_group": "FIRSTNAME",
    #         "score": 0.9992393255233765,
    #         "word": " 0",
    #         "start": 5,
    #         "end": 10
    #     },
    #     {
    #         "entity_group": "ETHEREUMADDRESS",
    #         "score": 0.9968568086624146,
    #         "word": "0x4eF4C3eCd2eDf372f0EaDFC3EaD841Bb9b4B9F82",
    #         "start": 45,
    #         "end": 87
    #     }
    # ]

    def detect_pii(self, text: str):
        # Define the endpoint URL
        url = "https://pv9staquijh8ucrz.us-east-1.aws.endpoints.huggingface.cloud"

        # Prepare headers and data payload for the HTTP request
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        data = {"inputs": text, "parameters": {"aggregation_strategy": "simple"}}

        # Make the HTTP POST request
        response = requests.post(url, json=data, headers=headers)

        # Default result if no PII detected
        result = {"pii_detected": False, "reason": []}

        # Check if the response contains detected PII entities
        if response.status_code == 200:
            pii_entities = response.json()
            if pii_entities:
                result["pii_detected"] = True
                result["reason"] = [
                    f"{entity['entity_group']} detected: {entity['word'].strip()}"
                    for entity in pii_entities
                ]
        else:
            raise Exception(f"Error occurred during PII detection: {response.text}")

        if not result["pii_detected"]:
            result["reason"] = "No PII detected"
        return result
