from typing import List, Optional, TypedDict
from athina.interfaces.athina import AthinaFilters
from athina.interfaces.data import DataPoint
from .base_loader import Loader


class RagasDataPoint(TypedDict):
    """Data point for a single RAG inference."""

    query: Optional[str]
    contexts: Optional[List[str]]
    response: Optional[str]
    expected_response: Optional[str]


class RagasLoader(Loader):
    """
    This class is a data loader for retrieval augmented generation (RAG) datasets.

    Attributes:
        col_query (str): The column name corresponding to the user's query.
        col_contexts (List[str]): The column name corresponding to the retrieved contexts.
        col_response (str): The column name corresponding to the response.
        col_expected_response (str): The column name corresponding to the expected response.
        raw_dataset (dict): The raw dataset as loaded from the source.
        processed_dataset (list): The processed dataset with queries, context, response and other attributes if present.
    """

    def __init__(
        self,
        col_query="query",
        col_contexts="contexts",
        col_response="response",
        col_expected_response="expected_response"
    ):
        """
        Initializes the loader with specified or default column names.
        """
        self.col_query = col_query
        self.col_contexts = col_contexts
        self.col_response = col_response
        self.col_expected_response = col_expected_response
        self._raw_dataset = {}
        self._processed_dataset: List[RagasDataPoint] = []

    def process(self) -> None:
        """
        Transforms the raw data into a structured format. Processes each entry from the raw dataset, and extracts attributes.

        Raises:
            TypeError: If any key is not of expected type.
        """
        for raw_instance in self._raw_dataset:
            # Check for the type of columns in raw_instance
            if self.col_query in raw_instance and not isinstance(raw_instance.get(self.col_query), str):
                raise TypeError(f"'{self.col_query}' is not of type string.")
            if self.col_contexts in raw_instance:
                if not isinstance(raw_instance.get(self.col_contexts), list):
                    raise TypeError(f"'{self.col_contexts}' is not of type list.")
                if not all(isinstance(element, str) for element in raw_instance.get(self.col_contexts)):
                    raise TypeError(f"Not all elements in '{self.col_context}' are of type string.")
            if self.col_response in raw_instance and not isinstance(raw_instance.get(self.col_response), str):
                raise TypeError(f"'{self.col_response}' is not of type string.")
            if self.col_expected_response in raw_instance and not isinstance(raw_instance.get(self.col_expected_response), str):
                raise TypeError(f"'{self.col_expected_response}' is not of type string.")
            

            # Create a processed instance
            processed_instance = {
                "query": raw_instance.get(self.col_query, None),
                "contexts": raw_instance.get(self.col_contexts, None),
                "response": raw_instance.get(self.col_response, None),
                "expected_response": raw_instance.get(self.col_expected_response, None)
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
        pass