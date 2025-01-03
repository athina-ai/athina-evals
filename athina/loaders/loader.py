from typing import List, Optional
from athina.interfaces.athina import AthinaFilters
from athina.interfaces.data import DataPoint as BaseDataPoint
from .base_loader import BaseLoader
from dataclasses import asdict
from athina.services.athina_api_service import AthinaApiService


class DataPoint(BaseDataPoint):
    """Data point for a single inference."""

    query: Optional[str]
    context: Optional[List[str]]
    response: Optional[str]
    expected_response: Optional[str]


class Loader(BaseLoader):
    """
    This class is a generic data loader for evals

    Attributes:
        col_query (str): The column name corresponding to the user's query.
        col_context (str): The column name corresponding to the retrieved context.
        col_response (str): The column name corresponding to the response.
        col_expected_response (str): The column name corresponding to the expected response.
        raw_dataset (dict): The raw dataset as loaded from the source.
        processed_dataset (list): The processed dataset with queries, context, response and other attributes if present.
    """

    def __init__(
        self,
        col_query="query",
        col_context="context",
        col_response="response",
        col_expected_response="expected_response",
    ):
        """
        Initializes the loader with specified or default column names.
        """
        self.col_query = col_query
        self.col_context = col_context
        self.col_response = col_response
        self.col_expected_response = col_expected_response
        self._raw_dataset = {}
        self._processed_dataset: List[DataPoint] = []

    def process(self) -> None:
        """
        Transforms the raw data into a structured format. Processes each entry from the raw dataset, and extracts attributes.
        """
        for raw_instance in self._raw_dataset:

            if self.col_query in raw_instance and not isinstance(
                raw_instance.get(self.col_query), str
            ):
                raise TypeError(f"'{self.col_query}' is not of type string.")
            if self.col_context in raw_instance:
                if not isinstance(raw_instance.get(self.col_context), list):
                    raise TypeError(f"'{self.col_context}' is not of type list.")
                if not all(
                    isinstance(element, str)
                    for element in raw_instance.get(self.col_context)
                ):
                    raise TypeError(
                        f"Not all elements in '{self.col_context}' are of type string."
                    )
            if self.col_response in raw_instance and not isinstance(
                raw_instance.get(self.col_response), str
            ):
                raise TypeError(f"'{self.col_response}' is not of type string.")
            if self.col_expected_response in raw_instance and not isinstance(
                raw_instance.get(self.col_expected_response), str
            ):
                raise TypeError(
                    f"'{self.col_expected_response}' is not of type string."
                )

            # Create a processed instance
            processed_instance = {
                "query": raw_instance.get(self.col_query, None),
                "context": raw_instance.get(self.col_context, None),
                "response": raw_instance.get(self.col_response, None),
                "expected_response": raw_instance.get(self.col_expected_response, None),
            }
            self._processed_dataset.append(processed_instance)

    def load_athina_inferences(
        self,
        filters: Optional[AthinaFilters] = None,
        limit: int = 10,
    ):
        """
        Load data from Athina API.
        By default, this will fetch the last 10 inferences from the API.
        """
        self._raw_dataset = AthinaApiService.fetch_inferences(
            filters=filters, limit=limit
        )
        for raw_dataset in self._raw_dataset:
            raw_dataset_dict = asdict(raw_dataset)

            context = (
                [str(raw_dataset_dict["context"])]
                if raw_dataset_dict["context"] is not None
                else None
            )
            processed_instance = {
                "query": raw_dataset_dict["user_query"],
                "context": context,
                "response": raw_dataset_dict["prompt_response"],
                "expected_response": raw_dataset_dict["expected_response"],
            }
            self._processed_dataset.append(processed_instance)
        return self._processed_dataset
