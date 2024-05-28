# Step to make a call to pinecone index to fetch relevent chunks
import pinecone
from typing import Optional, Union, Dict, Any
from athina.steps import Step
from jinja2 import Environment
from llama_index.vector_stores import QdrantVectorStore
from llama_index import VectorStoreIndex
from llama_index.retrievers import VectorIndexRetriever
import qdrant_client


class QdrantRetrieval(Step):
    """
    Step that makes a call to qdrant index to fetch relevant chunks.

    Attributes:
        collection_name: collection name in qdrant
        url: url of the qdrant server
        top_k: How many chunks to fetch.
        api_key: api key for the qdrant server
        input_column: column name in the input data
        env: jinja environment
    """

    collection_name: str
    url: str
    top_k: int
    api_key: str
    input_column: str
    env: Environment = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.qdrant_client = qdrant_client.QdrantClient(url=self.url, api_key=self.api_key)
        self.vector_store = QdrantVectorStore(client=self.qdrant_client, collection_name=self.collection_name)
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
