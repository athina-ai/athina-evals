import json
import os
from openai import OpenAI
from retrying import retry
from timeout_decorator import timeout
from athina_evals.keys import OpenAiApiKey

DEFAULT_MODEL = "gpt-3.5-turbo"
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
            raise Exception("Please provide an OpenAI API key")
        self.openai = OpenAI(api_key=openai_api_key)

    @timeout(30)
    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def chat_completion(
        self, messages, model=DEFAULT_MODEL, temperature=DEFAULT_TEMPERATURE
    ) -> str:
        """
        Fetches a completion response from OpenAI's ChatCompletion API based on the provided messages.
        """
        try:
            response = self.openai.chat.completions.create(
                model=model, messages=messages, temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in ChatCompletion: {e}")
            raise e
