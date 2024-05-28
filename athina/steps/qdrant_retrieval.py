# Step to make a call to pinecone index to fetch relevent chunks
import pinecone
from typing import Optional, Union, Dict, Any
from athina.steps import Step
from jinja2 import Environment
from athina.helpers.jinja_helper import PreserveUndefined
from llama_index.vector_stores import PineconeVectorStore
from llama_index import VectorStoreIndex
from llama_index.retrievers import VectorIndexRetriever


class QdrantRetrieval(Step):
    """
    Step that makes a call to qdrant index to fetch relevant chunks.

    Attributes:
        
    """

    index_name: str
    top_k: int
    metadata_filters: Any
    api_key: str
    input_column: str
    env: Environment = None
    namespace: Optional[str] = None
    environment: Optional[str] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        vector_store_args = {
            'api_key': self.api_key,
            'index_name': self.index_name
        }

        if self.environment is not None:
            vector_store_args['environment'] = self.environment

        if self.namespace is not None:
            vector_store_args['namespace'] = self.namespace

        self.vector_store = PineconeVectorStore(**vector_store_args)
        self.vector_index = VectorStoreIndex.from_vector_store(vector_store=self.vector_store)
        self.retriever = VectorIndexRetriever(index=self.vector_index, similarity_top_k=self.top_k)

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
            response = self.retriever.retrieve(input_text)
            return [node.get_content() for node in response]
        except Exception as e:
            return None
