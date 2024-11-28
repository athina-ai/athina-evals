import requests
import time
from typing import List, Optional
from ....keys import OpenAiApiKey
from ...base_evaluator import BaseEvaluator
from ....metrics.metric_type import MetricType
from ....evals.eval_type import FunctionEvalTypeId
from ....errors.exceptions import NoOpenAiApiKeyException
from ....interfaces.result import EvalResult, EvalResultMetric


class OpenAiContentModeration(BaseEvaluator):
    @property
    def name(self):
        return FunctionEvalTypeId.OPENAI_CONTENT_MODERATION.value

    @property
    def display_name(self):
        return "OpenAI Content Moderation"

    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.PASSED.value]

    @property
    def required_args(self):
        return ["text"]

    @property
    def examples(self):
        return None

    def __init__(self, open_ai_api_key: Optional[str] = None):
        if open_ai_api_key is None:
            if OpenAiApiKey.get_key() is None:
                raise NoOpenAiApiKeyException()
            self.open_ai_api_key = OpenAiApiKey.get_key()
        else:
            self.open_ai_api_key = open_ai_api_key

    def is_failure(self, content_moderation_response: dict) -> Optional[bool]:
        results = content_moderation_response.get("results", [])
        if results and len(results) > 0:
            # If the result is flagged, return True else False
            return bool(results[0].get("flagged", False))
        # Assuming when no results are returned, it is not a failure
        return False

    def get_reason(self, content_moderation_response: dict) -> Optional[str]:
        results = content_moderation_response.get("results", [])
        if results and len(results) > 0 and results[0].get("flagged", False):
            result = results[0]
            if results[0].get("flagged", False):
                flagged_categories = [
                    category
                    for category, flagged in result["categories"].items()
                    if flagged
                ]
                # Form a comma-separated string of flagged categories
                reason = ", ".join(flagged_categories)
                return f"The text was flagged in these categories: {reason}"
        return "The text was not flagged"

    def _evaluate(self, **kwargs) -> EvalResult:
        # Start timer
        start_time = time.perf_counter()
        self.validate_args(**kwargs)
        text = kwargs["text"]
        content_moderation_response = self.get_content_moderation_result(text)
        failure = self.is_failure(content_moderation_response)
        reason = self.get_reason(content_moderation_response)
        end_time = time.perf_counter()
        # Calculate runtime
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

    # EXAMPLE RESPONSE JSON
    # {
    #     "id": "modr-XXXXX",
    #     "model": "text-moderation-007",
    #     "results": [
    #         {
    #             "flagged": true,
    #             "categories": {
    #                 "sexual": false,
    #                 "hate": false,
    #                 "harassment": false,
    #                 "self-harm": false,
    #                 "sexual/minors": false,
    #                 "hate/threatening": false,
    #                 "violence/graphic": false,
    #                 "self-harm/intent": false,
    #                 "self-harm/instructions": false,
    #                 "harassment/threatening": true,
    #                 "violence": true
    #             },
    #             "category_scores": {
    #                 "sexual": 1.2282071e-6,
    #                 "hate": 0.010696256,
    #                 "harassment": 0.29842457,
    #                 "self-harm": 1.5236925e-8,
    #                 "sexual/minors": 5.7246268e-8,
    #                 "hate/threatening": 0.0060676364,
    #                 "violence/graphic": 4.435014e-6,
    #                 "self-harm/intent": 8.098441e-10,
    #                 "self-harm/instructions": 2.8498655e-11,
    #                 "harassment/threatening": 0.63055265,
    #                 "violence": 0.99011886
    #             }
    #         }
    #     ]
    # }

    def get_content_moderation_result(self, text: str):
        # Define the endpoint URL
        url = "https://api.openai.com/v1/moderations"
        # Prepare headers and data payload for the HTTP request
        headers = {
            "Authorization": f"Bearer {self.open_ai_api_key}",
            "Content-Type": "application/json",
        }
        data = {"input": text}
        # Make the HTTP POST request
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"Error occurred during OpenAI Content Moderation: {response}"
            )
