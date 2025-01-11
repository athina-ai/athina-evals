import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
from weaviate.client import WeaviateClient
from weaviate.collections.collection import Collection
from typing import Union, Dict, Any, List
from athina.steps import Step
from jinja2 import Environment
import time
import traceback


class WeaviateRetrieval(Step):
    """
    Step that makes a call to weaviate collection to fetch relevant chunks with similarity scores.

    Attributes:
    url: URL of the Weaviate instance.
    collection_name: Name of the Weaviate collection to query.
    key: Key to extract from the response objects.
    search_type: Type of search to perform (semantic_search, keyword_search, hybrid_search).
    limit: Maximum number of results to fetch.
    api_key: API key for the Weaviate server.
    openai_api_key: OpenAI Api Key.
    user_query: the query which will be sent to Weaviate
    env: Jinja environment.
    """

    url: str
    collection_name: str
    key: str
    search_type: str
    limit: int
    api_key: str
    openai_api_key: str
    user_query: str
    env: Environment = None
    _client: WeaviateClient = None
    _collection: Collection = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._client = weaviate.connect_to_weaviate_cloud(
            cluster_url=self.url,
            auth_credentials=Auth.api_key(self.api_key),
            headers={"X-OpenAI-Api-Key": self.openai_api_key},
            skip_init_checks=True,
        )
        self._collection = self._client.collections.get(self.collection_name)

    class Config:
        arbitrary_types_allowed = True

    def execute(self, input_data: Any) -> Union[Dict[str, Any], None]:
        """Makes a call to weaviate collection to fetch relevant chunks with scores"""
        start_time = time.perf_counter()

        if input_data is None:
            input_data = {}

        if not isinstance(input_data, dict):
            return self._create_step_result(
                status="error",
                data="Input data must be a dictionary.",
                start_time=start_time,
            )

        self.env = self._create_jinja_env()

        query_text = self.env.from_string(self.user_query).render(**input_data)

        if query_text is None:
            return self._create_step_result(
                status="error", data="Query text is Empty.", start_time=start_time
            )

        try:

            if self.search_type == "semantic_search":
                response = self._collection.query.near_text(
                    query=query_text,
                    limit=self.limit,
                    return_metadata=MetadataQuery.full(),
                )
            elif self.search_type == "keyword_search":
                response = self._collection.query.bm25(
                    query=query_text,
                    limit=self.limit,
                    return_metadata=MetadataQuery.full(),
                )
            elif self.search_type == "hybrid_search":
                response = self._collection.query.hybrid(
                    query=query_text,
                    limit=self.limit,
                    return_metadata=MetadataQuery.full(),
                )
            else:
                raise ValueError(f"Unsupported search type: {self.search_type}")
            print(response)
            results = []
            for obj in response.objects:
                if self.search_type == "semantic_search":
                    score = (
                        obj.metadata.certainty
                        if hasattr(obj.metadata, "certainty")
                        else None
                    )
                else:
                    score = (
                        obj.metadata.score if hasattr(obj.metadata, "score") else None
                    )

                result = {"text": obj.properties[self.key], "score": score}
                results.append(result)

            return self._create_step_result(
                status="success",
                data=results,
                start_time=start_time,
            )
        except Exception as e:
            traceback.print_exc()
            return self._create_step_result(
                status="error",
                data=str(e),
                start_time=start_time,
            )

    def close(self):
        """Closes the connection to the Weaviate client."""
        if self._client:
            self._client.close()
            self._client = None
