
from typing import Optional

from athina.metrics.metric_type import MetricType
import time
from typing import Optional
from athina.interfaces.result import EvalResult, EvalResultMetric
from athina.helpers.logger import logger
from athina.interfaces.athina import AthinaExperiment
from ..base_evaluator import BaseEvaluator
from .functions import operations

class FunctionEvaluator(BaseEvaluator):

    _function_name: str
    _function_arguments: dict
    _experiment: Optional[AthinaExperiment] = None
    _model: str

    """
    This evaluator runs the requested Function on the given data.
    """

    @property
    def _model(self):
        return ""
    
    @property
    def name(self):
        return self._function_name

    @property
    def display_name(self):
        return "Function evaluator"

    @property
    def metric_ids(self) -> str:
        return MetricType.PASSED.value

    @property
    def default_model(self):
        return None

    @property
    def default_function(self):
        return "contains_any"

    @property
    def default_function_arguments(self):
        return {}

    @property
    def required_args(self):
        return ["response"]

    @property
    def examples(self):
        return None

    def __init__(
        self,
        function_name: Optional[str] = None,
        function_arguments: Optional[dict] = None,
    ):
        if function_name is None:
            function_name = self.default_function
        if function_arguments is None:
            function_arguments = self.default_function_arguments
        if function_name not in operations.keys():
            raise ValueError(f"Unsupported function: {function_name}")
        else:
            self._function_name = function_name
            self._function_arguments = function_arguments


    def _evaluate(self, **kwargs) -> EvalResult:
        """
        Run the Function evaluator.
        """
        start_time = time.time()

        # Validate that correct args were passed
        self.validate_args(**kwargs)
        metrics = []
        try: 
            # Evaluate the dataset using Function
            operator = operations.get(self._function_name)
            response = operator(**kwargs, **self._function_arguments)
            metrics.append(EvalResultMetric(id=MetricType.PASSED.value, value=float(response["result"])))
            explanation = response['reason']

        except Exception as e:
            logger.error(f"Error occurred during eval: {e}")
            raise e

        end_time = time.time()
        eval_runtime_ms = int((end_time - start_time) * 1000)
        eval_result = EvalResult(
            name=self.name,
            display_name=self.display_name,
            data=kwargs,
            reason=explanation,
            runtime=eval_runtime_ms,
            model=self._model,
            metrics=metrics,
            failure=not response["result"] if response is not None and "result" in response else None,
        )
        return {k: v for k, v in eval_result.items() if v is not None}

