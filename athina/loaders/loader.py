from abc import ABC, abstractmethod
from typing import TypedDict, List
import json


class DataPoint(TypedDict):
    """Data point for a single inference."""

    response: str


class Loader(ABC):
    """Abstract base class for data loaders."""

    @abstractmethod
    def load(self, filename: str) -> List[DataPoint]:
        """Load data in the specified format."""
        pass

    @abstractmethod
    def process(self) -> List[DataPoint]:
        """Prepare dataset to be consumed by evaluators."""
        pass
