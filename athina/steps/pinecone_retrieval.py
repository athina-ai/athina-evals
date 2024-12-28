# Step to make a call to pinecone index to fetch relevent chunks
import pinecone
from typing import Optional, Union, Dict, Any

from pydantic import Field, PrivateAttr
from athina.steps import Step
from jinja2 import Environment
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever


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
        input_column: column name in the input data
        env: jinja environment
    """

    index_name: str
    top_k: int
    api_key: str
    input_column: str
    env: Environment = None
    metadata_filters: Optional[Dict[str, Any]] = None
    namespace: Optional[str] = None
    environment: Optional[str] = None
    _vector_store: PineconeVectorStore = PrivateAttr()
    _vector_index: VectorStoreIndex = PrivateAttr()
    _retriever: VectorIndexRetriever = PrivateAttr()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vector_store_args = {"api_key": self.api_key, "index_name": self.index_name}

        if self.environment is not None:
            vector_store_args["environment"] = self.environment

        if self.namespace is not None:
            vector_store_args["namespace"] = self.namespace

        self._vector_store = PineconeVectorStore(**vector_store_args)
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
        if input_data is None:
            input_data = {}

        if not isinstance(input_data, dict):
            raise TypeError("Input data must be a dictionary.")

        input_text = input_data.get(self.input_column, None)

        if input_text is None:
            return None

        try:
            response = self._retriever.retrieve(input_text)
            result = [node.get_content() for node in response]
            return {
                "status": "success",
                "data": result,
            }
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"Error during retrieval: {str(e)}")
            return {
                "status": "error",
                "data": str(e),
            }
