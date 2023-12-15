import json
import os
from openai import OpenAI
from retrying import retry
from timeout_decorator import timeout
from athina.keys import OpenAiApiKey
from athina.errors.exceptions import NoOpenAiApiKeyException

DEFAULT_TEMPERATURE = 0.0


class OpenAiService:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(OpenAiService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        openai_api_key = OpenAiApiKey.get_key()
        if openai_api_key is None:
            raise NoOpenAiApiKeyException()
        self.openai = OpenAI(api_key=openai_api_key)

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def chat_completion(self, messages, model, temperature=DEFAULT_TEMPERATURE) -> str:
        """
        Fetches response from OpenAI's ChatCompletion API.
        """
        try:
            response = self.openai.chat.completions.create(
                model=model, messages=messages, temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in ChatCompletion: {e}")
            raise e

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def json_completion(self, messages, model, temperature=DEFAULT_TEMPERATURE) -> str:
        """
        Fetches response from OpenAI's ChatCompletion API using JSON mode.
        """
        try:
            response = self.openai.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in ChatCompletion: {e}")
            raise e
