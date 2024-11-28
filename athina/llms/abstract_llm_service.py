from abc import ABC, abstractmethod

class AbstractLlmService(ABC):
    """
    Abstract class for different Language Learning Model (LLM) Providers.
    """

    @abstractmethod
    def embeddings(self, text: str) -> list:
        """
        Fetches embeddings for the given text. This method should be implemented by subclasses 
        to use the specific LLM provider's embeddings API.
        """
        raise NotImplementedError

    @abstractmethod
    def chat_completion(self, messages, model, **kwargs) -> str:
        """
        Fetches a chat completion response. This method should be implemented by subclasses 
        to interact with the specific LLM provider's chat completion API.
        """
        raise NotImplementedError

    @abstractmethod
    def chat_completion_json(self, messages, model, **kwargs) -> str:
        """
        Fetches a chat completion response in JSON format. This method should be implemented 
        by subclasses to interact with the specific LLM provider's chat completion API using JSON mode.
        """
        raise NotImplementedError

    @abstractmethod
    def json_completion(self, messages, model, **kwargs):
        """
        Helper method to be implemented by subclasses. This method should call either chat_completion or chat_completion_json.
        
        """
        raise NotImplementedError
