from typing import Optional, Union, Dict, Any

from pydantic import Field, PrivateAttr
from athina.steps import Step
from jinja2 import Environment
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
import time
import traceback


class PineconeRetrieval(Step):
    """
    Step that makes a call to pinecone index to fetch relevant chunks.

    Attributes:
        index_name: index name in pinecone
        namespace: namespace of the index.
        top_k: How many chunks to fetch.
        metadata_filters: filters to apply to metadata.
        environment: pinecone environment.
        api_key: api key for the pinecone server
        user_query: the query which will be sent to pinecone
        env: jinja environment
    """

    index_name: str
    top_k: int
    api_key: str
    user_query: str
    env: Environment = None
    metadata_filters: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None
    environment: Optional[str] = None
    text_key: Optional[str] = None  # Optional parameter for text key
    _vector_store: PineconeVectorStore = PrivateAttr()
    _vector_index: VectorStoreIndex = PrivateAttr()
    _retriever: VectorIndexRetriever = PrivateAttr()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize base vector store arguments
        vector_store_args = {"api_key": self.api_key, "index_name": self.index_name}
        # Add text_key only if specified by user
        if self.text_key:
            vector_store_args["text_key"] = self.text_key

        # Only add environment if it's provided
        if self.environment is not None:
            vector_store_args["environment"] = self.environment

        # Only add namespace if it's provided and not None
        if self.namespace:
            vector_store_args["namespace"] = self.namespace

        # Initialize vector store with filtered arguments
        self._vector_store = PineconeVectorStore(**vector_store_args)

        # Create vector index from store
        self._vector_index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store
        )

        # Initialize retriever with specified top_k
        self._retriever = VectorIndexRetriever(
            index=self._vector_index, similarity_top_k=self.top_k
        )

    class Config:
        arbitrary_types_allowed = True

    def execute(self, input_data: Any) -> Union[Dict[str, Any], None]:
        """Makes a call to pinecone index to fetch relevant chunks"""
        start_time = time.perf_counter()

        # Validate input data
        if input_data is None:
            input_data = {}

        if not isinstance(input_data, dict):
            return self._create_step_result(
                status="error",
                data="Input data must be a dictionary.",
                start_time=start_time,
            )

        # Create Jinja environment and render query
        self.env = self._create_jinja_env()
        query_text = self.env.from_string(self.user_query).render(**input_data)

        if not query_text:
            return self._create_step_result(
                status="error",
                data="Query text is Empty.",
                start_time=start_time,
            )

        try:
            # Perform retrieval
            response = self._retriever.retrieve(query_text)
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
            return self._create_step_result(
                status="success",
                data=result,
                start_time=start_time,
            )
        except Exception as e:
            traceback.print_exc()
            print(f"Error during retrieval: {str(e)}")
            return self._create_step_result(
                status="error",
                data=str(e),
                start_time=start_time,
            )
