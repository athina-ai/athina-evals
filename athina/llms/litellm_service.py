import litellm
from retrying import retry
from timeout_decorator import timeout
from athina.helpers.json import JsonHelper
from athina.keys import OpenAiApiKey
from athina.interfaces.model import Model
from athina.errors.exceptions import NoOpenAiApiKeyException
from .abstract_llm_service import AbstractLlmService
from typing import List, Dict, Any, Optional, Union, cast


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
    def chat_completion(
        self, messages: List[Dict[str, str]], model: str, **kwargs
    ) -> str:
        """
        Fetches response from Litellm's Completion API.
        """
        try:
            response = litellm.completion(
                api_key=self._api_key, model=model, messages=messages, **kwargs
            )
            if not response:
                raise ValueError("Empty response from LLM")

            # Convert response to dict if it's not already
            if not isinstance(response, dict):
                response = cast(Dict[str, Any], response.__dict__)

            # Handle different response formats
            if "choices" in response and response["choices"]:
                return str(response["choices"][0]["message"]["content"])
            elif "content" in response:
                return str(response["content"])
            else:
                return str(response)
        except Exception as e:
            print(f"Error in ChatCompletion: {e}")
            raise e

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def chat_completion_json(
        self, messages: List[Dict[str, str]], model: str, **kwargs
    ) -> str:
        raise NotImplementedError

    def json_completion(
        self, messages: List[Dict[str, str]], model: str, **kwargs
    ) -> str:
        raise NotImplementedError

    async def chat_stream_completion(
        self, messages: List[Dict[str, str]], model: str, **kwargs
    ) -> Any:
        """
        Fetches a streaming response from Litellm's Completion API.
        """
        try:
            response = litellm.completion(
                api_key=self._api_key,
                model=model,
                messages=messages,
                stream=True,
                **kwargs,
            )
            return response
        except Exception as e:
            print(f"Error in ChatStreamCompletion: {e}")
            raise e
