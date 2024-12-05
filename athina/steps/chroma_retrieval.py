# Step to make a call to chromadb collection to fetch relevant chunks

from typing import Union, Dict, Any, List, Optional, Literal
from athina.steps import Step
import chromadb
from chromadb.config import Settings
from enum import Enum
from jinja2 import Environment
from athina.helpers.jinja_helper import PreserveUndefined
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


class AuthType(str, Enum):
    TOKEN = "token"
    BASIC = "basic"
    NONE = "none"


class ChromaRetrieval(Step):
    """
    Step that retrieves documents from an existing Chroma collection.

    Attributes:
        host (str): The host of the Chroma server.
        port (int): The port of the Chroma server.
        collection_name (str): The name of the Chroma collection.
        limit (int): The maximum number of results to fetch.
        input_column (str): The column name in the input data.
        openai_api_key (str): The OpenAI API key.
        auth_type (str): The authentication type for the Chroma server (e.g., "token" or "basic").
        auth_credentials (str): The authentication credentials for the Chroma server.
    """

    host: str
    port: int
    collection_name: str
    key: str
    limit: int
    input_column: str
    openai_api_key: str
    auth_type: Optional[AuthType] = None
    auth_credentials: Optional[str] = None
    env: Environment = None
    _client: chromadb.Client = None
    _collection: chromadb.Collection = None
    _embedding_function = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        settings = None

        if self.auth_type == "none":
            settings = None
        elif self.auth_type is not None:
            auth_provider = {
                AuthType.TOKEN: "chromadb.auth.token_authn.TokenAuthClientProvider",
                AuthType.BASIC: "chromadb.auth.basic_authn.BasicAuthClientProvider",
            }.get(self.auth_type)

            if auth_provider and self.auth_credentials:
                settings = Settings(
                    chroma_client_auth_provider=auth_provider,
                    chroma_client_auth_credentials=self.auth_credentials,
                )
        else:
            settings = None

        self._client = chromadb.HttpClient(
            host=self.host, port=self.port, settings=settings
        )
        self._embedding_function = OpenAIEmbeddingFunction(api_key=self.openai_api_key)
        self._collection = self._client.get_collection(
            name=self.collection_name, embedding_function=self._embedding_function
        )
        # Create a custom Jinja2 environment with double curly brace delimiters and PreserveUndefined
        self.env = Environment(
            variable_start_string="{{",
            variable_end_string="}}",
            undefined=PreserveUndefined,
        )

    """Makes a call to chromadb collection to fetch relevant chunks"""

    def execute(self, input_data: Any) -> Union[Dict[str, Any], None]:
        if input_data is None or not isinstance(input_data, dict):
            return {"status": "error", "data": "Input data must be a dictionary."}

        query = input_data.get(self.input_column)
        if query is None:
            return {"status": "error", "data": "Input column not found."}

        try:
            if isinstance(query, list) and isinstance(query[0], float):
                response = self._collection.query(
                    query_embeddings=[query],
                    n_results=self.limit,
                    include=["documents", "metadatas", "distances"],
                )
            else:
                response = self._collection.query(
                    query_texts=[query],
                    n_results=self.limit,
                    include=["documents", "metadatas", "distances"],
                )

            return {"status": "success", "data": response["documents"][0]}
        except Exception as e:
            return {"status": "error", "data": str(e)}

    def close(self):
        if self._client:
            self._client = None
