from typing import List, Optional, TypedDict
from athina.interfaces.athina import AthinaFilters
from athina.interfaces.data import DataPoint
from .base_loader import Loader
from llama_index.indices.query.base import BaseQueryEngine
from dataclasses import asdict
from athina.services.athina_api_service import AthinaApiService


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
        col_expected_response="expected_response",
        query_engine: Optional[BaseQueryEngine] = None,
    ):
        """
        Initializes the loader with specified or default column names.
        """
        self.col_query = col_query
        self.col_contexts = col_contexts
        self.col_response = col_response
        self.col_expected_response = col_expected_response
        self.query_engine = query_engine
        self._raw_dataset = {}
        self._processed_dataset: List[RagasDataPoint] = []

    def _fetch_context_and_response_for_llama_index(self, query: str):
        """
        Fetches the context and response from the llama index query engine.
        """
        try:
            from llama_index.async_utils import run_async_tasks
        except ImportError:
            raise ImportError(
            "llama_index must be installed to use this function. "
            "Install it with `pip install llama_index`."
            )
        contexts = []
        query_engine_response = self.query_engine.query(query)
        response = query_engine_response.response
        for c in query_engine_response.source_nodes:
            text = c.node.get_text()
            contexts.append(text)

        return contexts, response

    def _generate_processed_instance_for_llama_index(self, raw_instance: dict) -> RagasDataPoint:
        """
        Generates a processed instance for the llama index query engine.
        """
        if self.col_query not in raw_instance:
            raise ValueError(f"'{self.col_query}' not found in provided data.")
        if self.col_query in raw_instance and not isinstance(raw_instance.get(self.col_query), str):
            raise TypeError(f"'{self.col_query}' is not of type string.")
        if self.col_expected_response in raw_instance and not isinstance(raw_instance.get(self.col_expected_response), str):
            raise TypeError(f"'{self.col_expected_response}' is not of type string.")

        contexts, response = self._fetch_context_and_response_for_llama_index(raw_instance.get(self.col_query))
        processed_instance = {
            "query": raw_instance.get(self.col_query),
            "contexts": contexts,
            "response": response,
            "expected_response": raw_instance.get(self.col_expected_response, None)
        }
        return processed_instance

    def _generate_processed_instance(self, raw_instance: dict) -> RagasDataPoint:
        """
        Generates a processed instance for the raw dataset.
        """
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
        return processed_instance

    def process(self) -> None:
        """
        Transforms the raw data into a structured format. Processes each entry from the raw dataset, and extracts attributes.

        Raises:
            TypeError: If any key is not of expected type.
        """
        if self.query_engine is not None:
            for raw_instance in self._raw_dataset:
                processed_instance = self._generate_processed_instance_for_llama_index(raw_instance)
                self._processed_dataset.append(processed_instance)
        else:
            for raw_instance in self._raw_dataset:
                processed_instance = self._generate_processed_instance(raw_instance)
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

            contexts = [str(raw_dataset_dict['context'])] if raw_dataset_dict['context'] is not None else None
            processed_instance = {
                "query": raw_dataset_dict['user_query'],
                "contexts": contexts,
                "response": raw_dataset_dict['prompt_response'],
                "expected_response": raw_dataset_dict['expected_response'] 
            }
            self._processed_dataset.append(processed_instance)
        return self._processed_dataset