from typing import List, Optional
from athina.interfaces.athina import AthinaFilters
from athina.interfaces.data import DataPoint
from .base_loader import BaseLoader
from athina.services.athina_api_service import AthinaApiService
from dataclasses import asdict



class ResponseLoader(BaseLoader):
    """
    This class is a data loader for evals that only evaluate the response.

    Attributes:
        col_response (str): The column name corresponding to the response.
        raw_dataset (dict): The raw dataset as loaded from the source.
        processed_dataset (list): The processed dataset with responses.
    """

    def __init__(
        self,
        col_response: str = "response",
        col_query: Optional[str] = "query",
        col_context: Optional[str] = "context",
        col_expected_response: Optional[str] = "expected_response",
    ):
        """
        Initializes the loader with specified or default column names.
        """
        self.col_response = col_response
        self.col_query = col_query
        self.col_context = col_context
        self.col_expected_response = col_expected_response
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
            if self.col_response not in raw_instance:
                raise KeyError(f"'{self.col_response}' not found in provided data.")
            if self.col_query in raw_instance and not isinstance(raw_instance.get(self.col_query), str):
                raise TypeError(f"'{self.col_query}' is not of type string.")
            if self.col_context in raw_instance and not isinstance(raw_instance.get(self.col_context), str):
                raise TypeError(f"'{self.col_context}' is not of type string.")
            if self.col_expected_response in raw_instance and not isinstance(raw_instance.get(self.col_expected_response), str):
                raise TypeError(f"'{self.col_expected_response}' is not of type string.")
            # Create a processed instance with mandatory fields
            processed_instance = {
                "response": raw_instance[self.col_response],
                "query": raw_instance.get(self.col_query, None),
                "context": raw_instance.get(self.col_context, None),
                "expected_response": raw_instance.get(self.col_expected_response, None),
            }
            # removing keys with None values
            processed_instance = {k: v for k, v in processed_instance.items() if v is not None}
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
                "response": raw_dataset_dict['prompt_response'],
            }
            self._processed_dataset.append(processed_instance)
        return self._processed_dataset