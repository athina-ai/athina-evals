# Step to make a call to pinecone index to fetch relevent chunks
from typing import Optional, Union, Dict, Any

from pydantic import PrivateAttr
from athina.steps import Step
from jinja2 import Environment
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
import qdrant_client
import time


class QdrantRetrieval(Step):
    """
    Step that makes a call to qdrant index to fetch relevant chunks.

    Attributes:
        collection_name: collection name in qdrant
        url: url of the qdrant server
        top_k: How many chunks to fetch.
        api_key: api key for the qdrant server
        user_query: the query which will be sent to qdrant
        env: jinja environment
    """

    collection_name: str
    url: str
    top_k: int
    api_key: str
    user_query: str
    env: Environment = None
    _qdrant_client: qdrant_client.QdrantClient = PrivateAttr()
    _vector_store: QdrantVectorStore = PrivateAttr()
    _vector_index: VectorStoreIndex = PrivateAttr()
    _retriever: VectorIndexRetriever = PrivateAttr()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._qdrant_client = qdrant_client.QdrantClient(
            url=self.url, api_key=self.api_key
        )
        self._vector_store = QdrantVectorStore(
            client=self._qdrant_client, collection_name=self.collection_name
        )
        self._vector_index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store
        )
        self._retriever = VectorIndexRetriever(
            index=self._vector_index, similarity_top_k=self.top_k
        )

    class Config:
        arbitrary_types_allowed = True

    def execute(self, input_data: Any) -> Union[Dict[str, Any], None]:
        """makes a call to pinecone index to fetch relevent chunks"""
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
            response = self._retriever.retrieve(query_text)
            if not response:
                print("No chunks retrieved for query text")
                return self._create_step_result(
                    status="success", data=[], start_time=start_time
                )
            result = [
                {
                    "text": node.get_content(),
                    "score": node.get_score(),
                }
                for node in response
            ]
            return self._create_step_result(
                status="success", data=result, start_time=start_time
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error during retrieval: {str(e)}")
            return self._create_step_result(
                status="error", data=str(e), start_time=start_time
            )
