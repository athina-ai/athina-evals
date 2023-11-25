from dataclasses import dataclass
from typing import TypedDict
from athina.loaders.loader import DataPoint


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
