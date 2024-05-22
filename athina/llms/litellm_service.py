import litellm
from retrying import retry
from timeout_decorator import timeout
from athina.helpers.json import JsonHelper
from athina.keys import OpenAiApiKey
from athina.interfaces.model import Model
from athina.errors.exceptions import NoOpenAiApiKeyException
from .abstract_llm_service import AbstractLlmService


class LitellmService(AbstractLlmService):
    _instance = None
    _api_key = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LitellmService, cls).__new__(cls)
        return cls._instance

    def __init__(self, api_key):
        self._api_key = api_key

    def embeddings(self, text: str) -> list:
        """
        Fetches response from OpenAI's Embeddings API.
        """
        raise NotImplementedError

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def chat_completion(self, messages, model, **kwargs) -> str:
        """
        Fetches response from Litellm's Completion API.
        """
        try:
            response = litellm.completion(api_key=self._api_key, model=model, messages=messages, **kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in ChatCompletion: {e}")
            raise e

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def chat_completion_json(self, messages, model, **kwargs) -> str:
        raise NotImplementedError

    def json_completion(self, messages, model, **kwargs):
        raise NotImplementedError
