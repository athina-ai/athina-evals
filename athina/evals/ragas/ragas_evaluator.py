
from typing import Optional
from athina.interfaces.athina import AthinaExperiment
from athina.interfaces.model import Model
import time
from typing import Optional
from athina.interfaces.result import EvalResult, EvalResultMetric
from athina.interfaces.model import Model
from athina.helpers.logger import logger

from ..base_evaluator import BaseEvaluator
from datasets import Dataset
from ragas.llms import LangchainLLM
from langchain.chat_models import ChatOpenAI
from ragas import evaluate


class RagasEvaluator(BaseEvaluator):
    _model: str
    _experiment: Optional[AthinaExperiment] = None

    def __init__(
        self,
        model: Optional[str] = None,
    ):
        if model is None:
            self._model = self.default_model
        elif not Model.is_supported(model):
            raise ValueError(f"Unsupported model: {model}")
        else:
            self._model = model

    def _validate_args(self, **kwargs) -> None:
        for arg in self.required_args:
            if arg not in kwargs:
                raise ValueError(f"Missing required argument: {arg}")

    def _evaluate(self, **kwargs) -> EvalResult:
        """
        Run the Ragas evaluator.
        """
        start_time = time.time()
        self._validate_args(**kwargs)
        try:
            chat_model = ChatOpenAI(model_name=self._model)
            self.ragas_metric.llm = LangchainLLM(llm=chat_model)
            # data = self.generate_data_to_evaluate(**kwargs)
            # print(data)
            data = {
                "contexts": [["France, in Western Europe, encompasses medieval cities, alpine villages and Mediterranean beaches. Paris, its capital, is famed for its fashion houses, classical art museums including the Louvre and monuments like the Eiffel Tower. The country is also renowned for its wines and sophisticated cuisine. Lascaux’s ancient cave drawings, Lyon’s Roman theater and the vast Palace of Versailles attest to its rich history"]],
                "question": ["What is the capital of France?"],
            }
            dataset = Dataset.from_dict(data)
            print(dataset)
            scores = evaluate(dataset, metrics=[self.ragas_metric])
            metric = EvalResultMetric(id=self.metric_id, value=scores[self.ragas_metric_name])
        except Exception as e:
            logger.error(f"Error occurred during eval: {e}")
            raise e

        end_time = time.time()
        eval_runtime_ms = int((end_time - start_time) * 1000)
        llm_eval_result = EvalResult(
            name=self.name,
            display_name=self.display_name,
            data=kwargs,
            failure=False,
            reason='',
            runtime=eval_runtime_ms,
            model=self._model,
            metric=metric,
        )
        return {k: v for k, v in llm_eval_result.items() if v is not None}

