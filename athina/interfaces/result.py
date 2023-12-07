from dataclasses import dataclass
from typing import TypedDict
from athina.interfaces.data import DataPoint


class LlmEvalResult(TypedDict):
    """
    A class to represent the result of an LLM evaluation.
    """

    name: str
    data: DataPoint
    failure: int
    reason: str
    runtime: int
    model: str


class EvalPerformanceMetrics(TypedDict):
    """
    A class to represent the performance metrics for an evaluation.
    """

    accuracy: float
    precision: float
    recall: float
    f1_score: float
