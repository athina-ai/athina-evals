from dataclasses import dataclass
from typing import TypedDict, List, Optional
from athina.interfaces.data import DataPoint


class LlmEvalResultMetric(TypedDict):
    """
    Represents the LLM evaluation result metric.
    """

    id: str
    value: float


class LlmEvalResult(TypedDict):
    """
    Represents the LLM evaluation result.
    """

    name: str
    display_name: str
    data: DataPoint
    failure: int
    reason: str
    runtime: int
    model: str
    metric: Optional[LlmEvalResultMetric]


class BatchRunResult(TypedDict):
    """
    Represents the result of a batch run of LLM evaluation.
    """

    eval_request_id: str
    eval_results: List[LlmEvalResult]


class EvalPerformanceReport(TypedDict):
    """
    Represents the performance metrics for an evaluation.
    """

    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    runtime: int
    dataset_size: int
