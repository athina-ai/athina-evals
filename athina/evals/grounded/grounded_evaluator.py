
from typing import Optional, List
from athina.evals.grounded.similarity import Comparator
from athina.metrics.metric_type import MetricType
import time
from typing import Optional
from athina.interfaces.result import EvalResult, EvalResultMetric
from athina.helpers.logger import logger
from athina.interfaces.athina import AthinaExperiment
from ..base_evaluator import BaseEvaluator

class GroundedEvaluator(BaseEvaluator):

    _comparator: Comparator
    _failure_threshold = None

    """
    This evaluator runs the requested grounded evaluator on the given data.
    """

    @property
    def _model(self):
        return None
    
    @property
    def name(self):
        return self._comparator.__class__.__name__

    @property
    def display_name(self):
        return self._comparator.__class__.__name__

    @property
    def metric_ids(self) -> List[str]:
        return [MetricType.SIMILARITY_SCORE.value]

    @property
    def examples(self):
        return None

    def __init__(
        self,
        comparator: Comparator = None,
        failure_threshold: Optional[float] = None,
    ):
        if comparator is None:
            raise ValueError(f"comparator is a required argument") 
        else:
            self._comparator = comparator
        if failure_threshold is not None:
            self._failure_threshold = failure_threshold

    def _process_kwargs(self, required_args, **kwargs):
        required_args_map = {
            key: "\n".join(kwargs[key]) if key == "context" and isinstance(kwargs[key], list) else kwargs[key] 
            for key in required_args
        }
        if len(required_args_map) == 2:
            values = list(required_args_map.values())
            if all(isinstance(value, str) for value in values):
                string1, string2 = values
                return string1, string2
            else:
                raise ValueError("Both arguments must be strings.")
        else:
            raise ValueError("Exactly two arguments are required.")

    def to_config(self):
        config = {
            "similarity_function": self._comparator.__class__.__name__,
        }
        if self._failure_threshold is not None:
            config["failure_threshold"] = self._failure_threshold
        return config
    
    
    def is_failure(self, score) -> Optional[bool]:
        return bool(score < self._failure_threshold) if self._failure_threshold is not None else None

    def _evaluate(self, **kwargs) -> EvalResult:
        """
        Run the Function evaluator.
        """
        start_time = time.perf_counter()

        # Validate that correct args were passed
        self.validate_args(**kwargs)
        metrics = []
        try: 
            string1, string2 = self._process_kwargs(self.required_args, **kwargs)
            # Calculate the similarity score using the comparator
            similarity_score = self._comparator.compare(string1, string2)
            metrics.append(EvalResultMetric(id=MetricType.SIMILARITY_SCORE.value, value=similarity_score))
            if self._failure_threshold is None:
                explanation = f"Successfully calculated similarity score of {similarity_score} using {self.display_name}"
            elif bool(similarity_score < self._failure_threshold):
                explanation = f"Evaluation failed as similarity score of {similarity_score} is below the failure threshold of {self._failure_threshold} using {self.display_name}"
            else:
                explanation = f"Evaluation succeeded as similarity score of {similarity_score} is above the failure threshold of {self._failure_threshold} using {self.display_name}"

            failure = self.is_failure(similarity_score)
        except Exception as e:
            logger.error(f"Error occurred during eval: {e}")
            raise e

        end_time = time.perf_counter()
        eval_runtime_ms = int((end_time - start_time) * 1000)
        eval_result = EvalResult(
            name=self.name,
            display_name=self.display_name,
            data=kwargs,
            reason=explanation,
            runtime=eval_runtime_ms,
            model=None,
            metrics=metrics,
            failure=failure
        )
        return {k: v for k, v in eval_result.items() if v is not None}

