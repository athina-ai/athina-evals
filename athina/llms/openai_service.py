from openai import OpenAI
from retrying import retry
from timeout_decorator import timeout
from athina.helpers.json import JsonHelper
from athina.keys import OpenAiApiKey
from athina.interfaces.model import Model
from athina.errors.exceptions import NoOpenAiApiKeyException
from .abstract_llm_service import AbstractLlmService
import json
import time 

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

    def embeddings(self, text: str, model: str) -> list:
        """
        Fetches response from OpenAI's Embeddings API.
        """
        try:
            response = self.openai.embeddings.create(
                model=model, input=text, encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error in Embeddings: {e}")
            raise e

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def chat_completion(self, messages, model, **kwargs) -> str:
        """
        Fetches response from OpenAI's ChatCompletion API.
        """
        if 'temperature' not in kwargs:
            kwargs['temperature'] = DEFAULT_TEMPERATURE
        try:
            start_time = time.time()
            response = self.openai.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            end_time = time.time()
            completion_time = (end_time - start_time) * 1000
            metadata = json.dumps({
                "usage": {
                    "completion_tokens": response.usage.completion_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "response_time": completion_time
            })
            if response.choices[0].finish_reason == 'tool_calls':
                tool_calls = [call.model_dump() for call in response.choices[0].message.tool_calls]
                tool_calls_data = [{"arguments": call["function"]["arguments"], "name": call["function"]["name"]} for call in tool_calls]
                return {"value": json.dumps(tool_calls_data), "metadata": metadata}
            else:
                prompt_response = response.choices[0].message.content
                
                if not prompt_response:
                    if response.choices[0].message.tool_calls:
                        tool_calls = [call.model_dump() for call in response.choices[0].message.tool_calls]
                        tool_calls_data = [{"arguments": call["function"]["arguments"], "name": call["function"]["name"]} for call in tool_calls]
                        return {"value": json.dumps(tool_calls_data), "metadata": metadata}
                    else:
                        return {"value": json.dumps(response.choices[0].message.__dict__), "metadata": metadata}
                return {"value": prompt_response, "metadata": metadata}
        except Exception as e:
            print(f"Error in ChatCompletion: {e}")
            raise e

    @retry(stop_max_attempt_number=3, wait_fixed=2000)
    def chat_completion_json(self, messages, model, **kwargs) -> str:
        """
        Fetches response from OpenAI's ChatCompletion API using JSON mode.
        """
        if 'temperature' not in kwargs:
            kwargs['temperature'] = DEFAULT_TEMPERATURE
        try:
            response = self.openai.chat.completions.create(
                model=model,
                messages=messages,
                response_format={"type": "json_object"},
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error in ChatCompletion: {e}")
            raise e

    def json_completion(self, messages, model, **kwargs):
        """
        Fetches response from OpenAI's ChatCompletion API using JSON mode.
        """
        if 'temperature' not in kwargs:
            kwargs['temperature'] = DEFAULT_TEMPERATURE
        try:
            if Model.supports_json_mode(model):
                chat_completion_response = self.chat_completion_json(
                    model=model,
                    messages=messages,
                    **kwargs,
                )
            else:
                chat_completion_response = self.chat_completion(
                    model=model,
                    messages=messages,
                    **kwargs,
                )
                chat_completion_response = chat_completion_response["value"]

            # Extract JSON object from LLM response
            return JsonHelper.extract_json_from_text(chat_completion_response)

        except Exception as e:
            print(f"Error in ChatCompletion: {e}")
            raise e
