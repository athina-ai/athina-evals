from dataclasses import dataclass
from typing import TypedDict


class LlmEvalResult(TypedDict):
    """
    A class to represent the result of an LLM evaluation.
    """

    failure: int
    reason: str
