from openai import OpenAI
from retrying import retry
from timeout_decorator import timeout
from athina.helpers.json import JsonHelper
from athina.keys import OpenAiApiKey
from athina.interfaces.model import Model
from athina.errors.exceptions import NoOpenAiApiKeyException
from .abstract_llm_service import AbstractLlmService

DEFAULT_TEMPERATURE = 0.0


class OpenAiService(AbstractLlmService):
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

    def embeddings(self, text: str) -> list:
        """
        Fetches response from OpenAI's Embeddings API.
        """
        try:
            response = self.openai.embeddings.create(
                model="text-embedding-ada-002",
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error in Embeddings: {e}")
            raise e

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
    def chat_completion_json(self, messages, model, temperature=DEFAULT_TEMPERATURE) -> str:
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

    def json_completion(self, messages, model, temperature=DEFAULT_TEMPERATURE):
        """
        Fetches response from OpenAI's ChatCompletion API using JSON mode.
        """
        try:
            if Model.supports_json_mode(model):
                chat_completion_response = self.chat_completion_json(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )
            else:
                chat_completion_response = self.chat_completion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                )

            # Extract JSON object from LLM response
            return JsonHelper.extract_json_from_text(chat_completion_response)
        
        except Exception as e:
            print(f"Error in ChatCompletion: {e}")
            raise e