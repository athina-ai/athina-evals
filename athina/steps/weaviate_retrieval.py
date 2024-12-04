# Step to make a call to weaviate collection to fetch relevant chunks
import weaviate
from weaviate.classes.init import Auth
from weaviate.client import WeaviateClient
from weaviate.collections.collection import Collection
from typing import Union, Dict, Any
from athina.steps import Step
from jinja2 import Environment


class WeaviateRetrieval(Step):
    """
    Step that makes a call to weaviate collection to fetch relevant chunks.

    Attributes:
    url: URL of the Weaviate instance.
    collection_name: Name of the Weaviate collection to query.
    key: Key to extract from the response objects.
    search_type: Type of search to perform (semantic_search, keyword_search, hybrid_search).
    limit: Maximum number of results to fetch.
    api_key: API key for the Weaviate server.
    openai_api_key: OpenAI Api Key.
    input_column: Column name in the input data.
    env: Jinja environment.
    """

    url: str
    collection_name: str
    key: str
    search_type: str
    limit: int
    api_key: str
    openai_api_key: str
    input_column: str
    env: Environment = None
    _client: WeaviateClient = None
    _collection: Collection = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.url,
            auth_credentials=Auth.api_key(self.api_key),
            headers={"X-OpenAI-Api-Key": self.openai_api_key},
        )
        self._collection = self._client.collections.get(self.collection_name)

    class Config:
        arbitrary_types_allowed = True

    def execute(self, input_data: Any) -> Union[Dict[str, Any], None]:
        """Makes a call to weaviate collection to fetch relevant chunks"""

        if input_data is None:
            input_data = {}

        if not isinstance(input_data, dict):
            raise TypeError("Input data must be a dictionary.")

        input_text = input_data.get(self.input_column, None)

        if input_text is None:
            return {"status": "error", "data": "Input column not found."}

        try:
            if self.search_type == "semantic_search":
                response = self._collection.query.near_text(
                    query=input_text, limit=self.limit
                )
            elif self.search_type == "keyword_search":
                response = self._collection.query.bm25(
                    query=input_text, limit=self.limit
                )
            elif self.search_type == "hybrid_search":
                response = self._collection.query.hybrid(
                    query=input_text, limit=self.limit
                )
            else:
                raise ValueError(f"Unsupported search type: {self.search_type}")

            results = []
            for obj in response.objects:
                results.append(obj.properties[self.key])

            return {
                "status": "success",
                "data": results,
            }
        except Exception as e:
            return {
                "status": "error",
                "data": str(e),
            }

    def close(self):
        """Closes the connection to the Weaviate client."""
        if self._client:
            self._client.close()
            self._client = None
