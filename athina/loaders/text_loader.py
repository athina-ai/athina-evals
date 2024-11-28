from .base_loader import BaseLoader
from typing import List, Optional
from athina.interfaces.athina import AthinaFilters
from athina.interfaces.data import DataPoint
from athina.services.athina_api_service import AthinaApiService
from dataclasses import asdict


class TextLoader(BaseLoader):
    """
    This class is a data loader for evals that only evaluate the response.

    Attributes:
        col_text (str): The column name corresponding to the response.
        raw_dataset (dict): The raw dataset as loaded from the source.
        processed_dataset (list): The processed dataset with responses.
    """

    def __init__(
        self,
        col_text: str = "text",
        col_expected_text: str = "expected_text",
    ):
        """
        Initializes the loader with specified or default column names.
        """
        self.col_text = col_text
        self.col_expected_text = col_expected_text
        self._raw_dataset = {}
        self._processed_dataset: List[DataPoint] = []

    def process(self) -> None:
        """
        Transforms the raw data into a structured format. Processes each entry from the raw dataset, and extracts attributes.

        Raises:
            KeyError: If mandatory columns (response) are missing in the raw dataset.
        """
        for raw_instance in self._raw_dataset:
            # Check for mandatory columns in raw_instance
            if self.col_text not in raw_instance:
                raise KeyError(f"'{self.col_text}' not found in provided data.")
            # Create a processed instance with mandatory fields
            processed_instance = {
                "text": raw_instance[self.col_text],
            }
            if self.col_expected_text in raw_instance:
                processed_instance["expected_text"] = raw_instance[
                    self.col_expected_text
                ]
            # removing keys with None values
            processed_instance = {
                k: v for k, v in processed_instance.items() if v is not None
            }
            # Store the results
            self._processed_dataset.append(processed_instance)

    def load_athina_inferences(
        self,
        filters: Optional[AthinaFilters] = None,
        limit: Optional[int] = None,
    ):
        """
        Load data from Athina API.
        """
        self._raw_dataset = AthinaApiService.fetch_inferences(
            filters=filters, limit=limit
        )
        for raw_dataset in self._raw_dataset:
            raw_dataset_dict = asdict(raw_dataset)
            processed_instance = {
                "text": raw_dataset_dict["prompt_response"],
            }
            self._processed_dataset.append(processed_instance)
        return self._processed_dataset
