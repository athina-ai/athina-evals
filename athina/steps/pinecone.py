# Step to make a call to pinecone index to fetch relevent chunks
import json
import pinecone
from typing import Union, Dict, Any, Iterable, Optional
import requests
from athina.llms.abstract_llm_service import AbstractLlmService
from athina.steps import Step
from jinja2 import Environment
from athina.helpers.jinja_helper import PreserveUndefined


class Pinecone(Step):
    """
    Step that makes a call to pinecone index to fetch relevent chunks.

    Attributes:
        index: index name in pinecone
        namespace: namespace of the index.
        top_k: How many chunks to fetch.
        metadata_filters: filters to apply to metadata.
    """

    index: str
    namespace: str
    top_k: int
    metadata_filters: Any
    embedding_provider: str
    embedding_model: str
    api_key: str
    environment: str
    llm_service: AbstractLlmService = None
    query_parameter: str
    env: Environment = None

    class Config:
        arbitrary_types_allowed = True

    def execute(self, input_data: Any) -> Union[Dict[str, Any], None]:
        """makes a call to pinecone index to fetch relevent chunks"""

        if input_data is None:
            input_data = {}

        if not isinstance(input_data, dict):
            raise TypeError("Input data must be a dictionary.")

        # Create a custom Jinja2 environment with double curly brace delimiters and PreserveUndefined
        self.env = Environment(
            variable_start_string='{{', 
            variable_end_string='}}',
            undefined=PreserveUndefined
        )

        if self.metadata_filters is not None:
            pass # Implement metadata filters

        try:
            pinecone.init(api_key=self.api_key, environment=self.environment)
            index = pinecone.index(self.index)
        except Exception as e:
            print(f"Error in initializing pinecone: {e}")
            raise e
        
        try:
            openai_embedding = self.get_embedding(input_data[self.query_parameter])
            result = index.query(
                vector=openai_embedding, top_k=self.top_k, include_values=True, include_metadata=True
            )
            if result:
                chunks = [x['metadata']['chunk'] for x in result["matches"]]
                return chunks
        except Exception as e:
            print(f"Error in querying pinecone: {e}")
            return None

    def get_embedding(self, text: str) -> Dict[str, Any]:
        """Get the embedding of the given text using the specified embedding provider and model."""
        if self.embedding_provider == "openai":
            return self.llm_service.embeddings(text, self.embedding_model)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")
