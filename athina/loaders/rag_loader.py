from typing import List, Optional
from athina.interfaces.athina import AthinaFilters
from athina.interfaces.data import DataPoint
from .base_loader import Loader
from dataclasses import asdict
from athina.services.athina_api_service import AthinaApiService


class RagDataPoint(DataPoint):
    """Data point for a single RAG inference."""

    query: str
    context: str
    response: str


class RagLoader(Loader):
    """
    This class is a data loader for retrieval augmented generation (RAG) datasets.

    Attributes:
        col_query (str): The column name corresponding to the user's query.
        col_context (str): The column name corresponding to the retrieved context.
        col_response (str): The column name corresponding to the response.
        raw_dataset (dict): The raw dataset as loaded from the source.
        processed_dataset (list): The processed dataset with queries, context, response and other attributes if present.
    """

    def __init__(
        self,
        col_query="query",
        col_context="context",
        col_response="response",
    ):
        """
        Initializes the loader with specified or default column names.
        """
        self.col_query = col_query
        self.col_context = col_context
        self.col_response = col_response
        self._raw_dataset = {}
        self._processed_dataset: List[RagDataPoint] = []

    def process(self) -> None:
        """
        Transforms the raw data into a structured format. Processes each entry from the raw dataset, and extracts attributes.

        Raises:
            KeyError: If mandatory columns (query, context or response) are missing in the raw dataset.
        """
        for raw_instance in self._raw_dataset:
            # Check for mandatory columns in raw_instance
            if self.col_query not in raw_instance:
                raise KeyError(f"'{self.col_query}' not found in provided data.")
            if self.col_context not in raw_instance:
                raise KeyError(f"'{self.col_context}' not found in provided data.")
            if self.col_response not in raw_instance:
                raise KeyError(f"'{self.col_response}' not found in provided data.")
            

            # Create a processed instance with mandatory fields
            processed_instance = {
                "query": raw_instance[self.col_query],
                "context": raw_instance[self.col_context],
                "response": raw_instance[self.col_response],
            }


            # Store the results
            self._processed_dataset.append(processed_instance)

    def load_athina_inferences(
        self,
        filters: Optional[AthinaFilters] = None,
        limit: int = 10,
        context_key: Optional[str] = None,
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
            processed_instance = {
                "query": raw_dataset_dict['user_query'],
                "context": raw_dataset_dict['context'],
                "response": raw_dataset_dict['prompt_response'],
            }

            # Store the results
            self._processed_dataset.append(processed_instance)
        return self._processed_dataset
          