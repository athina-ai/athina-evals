import math
import time
from abc import abstractmethod
from typing import Optional, Any

from athina.interfaces.athina import AthinaExperiment
from athina.interfaces.custom_model_config import CustomModelConfig
from athina.interfaces.model import Model
from athina.interfaces.result import EvalResult, EvalResultMetric
from athina.helpers.logger import logger
from ..base_evaluator import BaseEvaluator
from datasets import Dataset
from langchain_openai.chat_models import ChatOpenAI, AzureChatOpenAI
from ragas.llms import LangchainLLM
from ragas import evaluate


class RagasEvaluator(BaseEvaluator):
    _model: str
    _provider: Optional[str] = None
    _config: Optional[CustomModelConfig] = None
    _api_key: Optional[str]
    _experiment: Optional[AthinaExperiment] = None
    _failure_threshold: Optional[float] = None

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        provider: Optional[str] = "openai",  # Default provider set to 'openai'
        config: Optional[CustomModelConfig] = None,
        failure_threshold: Optional[float] = None,
    ):
        self._model = model
        self._provider = provider
        self._api_key = api_key
        self._config = config

        if failure_threshold is not None:
            self._failure_threshold = failure_threshold

    @property
    def default_model(self) -> str:
        return Model.GPT35_TURBO.value

    def generate_data_to_evaluate(self, **kwargs):
        pass

    @abstractmethod
    def ragas_metric(self) -> Any:
        pass

    @property
    def grade_reason(self) -> str:
        raise NotImplementedError

    def _get_model(self):
        if self._provider == "openai":
            return ChatOpenAI(model_name=self._model, api_key=self._api_key)
        elif self._provider == "azure":
            # Extracting azure configuration from completion_config
            azure_endpoint = None
            api_version = None
            for item in self._config.completion_config:
                if "api_base" in item:
                    azure_endpoint = item["api_base"]
                if "api_version" in item:
                    api_version = item["api_version"]

            if azure_endpoint is None or api_version is None:
                raise ValueError(
                    "Azure configuration is missing required fields 'api_base' or 'api_version'"
                )

            return AzureChatOpenAI(
                api_version=api_version,
                azure_endpoint=azure_endpoint,
                azure_deployment=self._model,
                api_key=self._api_key,
            )
        else:
            raise ValueError(f"Unsupported provider: {self._provider}")

    def _evaluate(self, **kwargs) -> EvalResult:
        """
        Run the Ragas evaluator.
        """
        start_time = time.time()
        self.validate_args(**kwargs)
        metrics = []
        try:
            self.ragas_metric.llm = LangchainLLM(llm=self._get_model())
            data = self.generate_data_to_evaluate(**kwargs)
            dataset = Dataset.from_dict(data)
            scores = evaluate(dataset, metrics=[self.ragas_metric])
            metric_value = scores[self.ragas_metric_name]
            if isinstance(metric_value, (int, float)) and not math.isnan(metric_value):
                metrics.append(
                    EvalResultMetric(id=self.metric_ids[0], value=metric_value)
                )
            else:
                logger.warn(f"Invalid metric value: {metric_value}")

            failure = self.is_failure(score=metric_value)
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
            reason=self.grade_reason,
            runtime=eval_runtime_ms,
            model=self._model,
            metrics=metrics,
        )
        return {k: v for k, v in llm_eval_result.items() if v is not None}
