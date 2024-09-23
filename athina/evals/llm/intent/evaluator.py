import time
from typing import List, Tuple, Optional

from athina.interfaces.result import EvalResult, EvalResultMetric, DatapointFieldAnnotation
from athina.helpers.logger import logger
from ....metrics.metric_type import MetricType
from ..llm_evaluator import LlmEvaluator
from .prompt import INTENT_EVAL_PROMPT_CONCISE_SYSTEM, INTENT_EVAL_PROMPT_CONCISE_USER

class Intent(LlmEvaluator):
    
    def __init__(self, **kwargs):
        self.model_name = "HUGGINGFACE_META_LLAMA_3_70B"
        super().__init__(**kwargs, system_message_template=INTENT_EVAL_PROMPT_CONCISE_SYSTEM,
            user_message_template=INTENT_EVAL_PROMPT_CONCISE_USER,)

    @property
    def _model(self):
        return self.model_name

    @property
    def name(self):
        return "Intent"
    
    @property
    def display_name(self):
        return "Intent"
    
    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.PASSED.value]
    
    @property
    def default_function_arguments(self):
        return {}
    
    @property
    def required_args(self):
        # expects an array of strings from ["query", "context", "response", "expected_response", "text"]
        return ["query", "response"]
    
    @property
    def examples(self):
        pass

    def is_failure(self, result: bool) -> bool:
        return not(bool(result))
    
    
    def _evaluate(self, **kwargs) -> EvalResult:
        start_time = time.time()
        self.validate_args(**kwargs)
        messages = self._prompt_messages(**kwargs)

        chat_completion_response_json: dict = self.llm_service.json_completion(
            model=self._model,
            messages=messages,
            temperature=self.TEMPERATURE,
        )

        malicious_keywords = ["malicious", "illegal", "harm", "harmful", "unlawful", "hurt", "pain", "hate"]
        for keyword in malicious_keywords:
            if keyword.lower() in chat_completion_response_json["result"].lower():
                self.label = "malicious"
        self.label = "normal"

        metrics = []

        try:
            result = chat_completion_response_json["result"]
            failure = self.is_failure(result)
            passed_value = 1 - float(failure)
            metrics.append(EvalResultMetric(id=MetricType.PASSED.value, value=passed_value))
            label: str = self.label

        except Exception as e:
            logger.error(f"Error occurred during eval: {e}")
            raise e

        end_time = time.time()
        eval_runtime_ms = int((end_time - start_time) * 1000)
        llm_eval_result = EvalResult(
            name=self.name,
            display_name=self.display_name,
            data=kwargs,
            failure=failure,
            runtime=eval_runtime_ms,
            model=self._model,
            metrics=metrics,
            #label = self.label
        )
        return {k: v for k, v in llm_eval_result.items() if v is not None}