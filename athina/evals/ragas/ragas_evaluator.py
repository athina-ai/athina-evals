import math
from abc import abstractmethod
from typing import Optional
from athina.interfaces.athina import AthinaExperiment
from athina.interfaces.model import Model
import time
from typing import Optional, Any, List
from athina.interfaces.result import EvalResult, EvalResultMetric
from athina.interfaces.model import Model
from athina.helpers.logger import logger
from ..base_evaluator import BaseEvaluator
from datasets import Dataset
from langchain_openai.chat_models import ChatOpenAI
from ragas.llms import LangchainLLM
from ragas import evaluate
from athina.keys import OpenAiApiKey


class RagasEvaluator(BaseEvaluator):
    _model: str
    _openai_api_key: Optional[str]
    _experiment: Optional[AthinaExperiment] = None
    _failure_threshold: Optional[float] = None

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        model: Optional[str] = None,
        failure_threshold: Optional[float] = None
    ):
        if model is None:
            self._model = self.default_model
        else:
            self._model = model
        
        if openai_api_key is None:
            self._openai_api_key = OpenAiApiKey.get_key()
        else:
            self._openai_api_key = openai_api_key

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
        return ChatOpenAI(model_name=self._model, api_key=self._openai_api_key)

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
                metrics.append(EvalResultMetric(id=self.metric_ids[0], value=metric_value))
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


