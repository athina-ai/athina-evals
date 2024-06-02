from abc import ABC, abstractmethod
from enum import Enum
from typing import List
import json
from athina.interfaces.data import DataPoint


class LoadFormat(Enum):
    """Supported load formats."""

    JSON = "json"
    DICT = "dict"
    ATHINA = "athina"


class BaseLoader(ABC):
    """Abstract base class for data loaders."""

    @property
    def processed_dataset(self) -> List[DataPoint]:
        """
        Returns the processed dataset.
        """
        return self._processed_dataset

    @property
    def raw_dataset(self):
        """
        Returns the raw dataset.
        """
        return self._raw_dataset

    @abstractmethod
    def process(self) -> List[DataPoint]:
        """Prepare dataset to be consumed by evaluators."""
        pass

    def load(self, format: str, **kwargs) -> List[DataPoint]:
        """
        Loads data based on the format specified.
        """
        if format == LoadFormat.JSON.value:
            return self.load_json(**kwargs)
        elif format == LoadFormat.DICT.value:
            return self.load_dict(**kwargs)
        elif format == LoadFormat.ATHINA.value:
            return self.load_athina_inferences(**kwargs)
        else:
            raise NotImplementedError("This file format has not been supported yet.")

    def load_json(self, filename: str) -> List[DataPoint]:
        """
        Loads and processes data from a JSON file.

        Raises:
            FileNotFoundError: If the specified JSON file is not found.
            json.JSONDecodeError: If there's an issue decoding the JSON.
        """
        try:
            with open(filename, "r") as f:
                self._raw_dataset = json.load(f)
                self.process()
                return self._processed_dataset
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading JSON: {e}")

    def load_dict(self, data: list) -> List[DataPoint]:
        """
        Loads and processes data from a list of dictionaries.
        """
        self._raw_dataset = data
        self.process()
        return self._processed_dataset

    @abstractmethod
    def load_athina_inferences(self, data: dict) -> List[DataPoint]:
        """
        Loads and processes data from a dictionary of Athina inferences.
        """
        pass
