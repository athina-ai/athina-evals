import os
import openai
import time
import traceback
import json
from typing import Optional, List
from athina_evals.helpers.logger import logger
from athina_logger.inference_logger import InferenceLogger
from athina_logger.api_key import AthinaApiKey
from athina_logger.exception.custom_exception import CustomException


class OpenAiLlmService:
    """
    A class to interact with OpenAI's ChatCompletion API.

    Attributes:
    - model (str): Model to use for completions, default is "gpt-3.5-turbo".
    - open_ai_key (str): API key for OpenAI.
    - metadata (dict): Metadata to be logged to Athina.
    """

    def __init__(
        self,
        model: str,
        metadata: Optional[dict] = None,
    ):
        """
        Initializes the OpenAICompletion with the provided settings.
        """
        # Setting instance attributes based on provided parameters or defaults
        self.model = model
        self.metadata = metadata
        athina_api_key = os.environ.get("ATHINA_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        AthinaApiKey.set_api_key(athina_api_key)
        openai.api_key = self.openai_api_key

    def chat_completion(
        self,
        messages: List[dict],
        temperature: float = 0,
        max_tokens: int = 2000,  # max tokens in completion response
        retry_count: int = 0,
        **kwargs,
    ):
        """
        Fetches a completion response from OpenAI's ChatCompletion API based on the provided messages.
        """
        try:
            # Attempting to fetch a response from OpenAI
            start_time = time.time()
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            end_time = time.time()
            response_time_ms = int((end_time - start_time) * 1000)

            # Logging the response to Athina
            if self.metadata is None:
                environment = None
                prompt_slug = None
                customer_id = None
                customer_user_id = None
                external_reference_id = None
                session_id = None
            else:
                environment = (
                    self.metadata["environment"]
                    if self.metadata["environment"] is not None
                    else "default"
                )
                prompt_slug = (
                    self.metadata["prompt_slug"]
                    if self.metadata["prompt_slug"] is not None
                    else "default"
                )
                customer_id = self.metadata["customer_id"]
                customer_user_id = self.metadata["customer_user_id"]
                external_reference_id = self.metadata["external_reference_id"]
                session_id = self.metadata["session_id"]

            if AthinaApiKey.get_api_key() is not None:
                InferenceLogger.log_open_ai_chat_response(
                    prompt_slug=prompt_slug,
                    messages=messages,
                    model=self.model,
                    completion=response,
                    context=None,
                    response_time=response_time_ms,
                    customer_id=customer_id,
                    customer_user_id=customer_user_id,
                    external_reference_id=external_reference_id,
                    session_id=session_id,
                    environment=environment,
                )

        except openai.error.RateLimitError as e:
            print("RateLimitError", e)
            # Calculate the wait time using exponential backoff
            base_wait_time = 15
            max_retries = 3
            wait_time = base_wait_time * (2**retry_count)

            # Wait for the calculated wait time
            time.sleep(wait_time)

            if retry_count < max_retries:
                return self.chat_completion(
                    messages, temperature, max_tokens, retry_count + 1
                )
            else:
                print("Max retries reached - unable to complete OpenAI request")
                raise e
        # Example for improved error handling
        except openai.error.AuthenticationError as e:
            error_message = f"AuthenticationError: Please pass a valid OpenAI key. Original error: {e}\nTraceback: {traceback.format_exc()}"
            logger.error(error_message)
            raise e
        except openai.error.Timeout as e:
            error_message = (
                f"OpenAI Timeout Error: {e}\n Traceback: {traceback.format_exc()}"
            )
            logger.error(error_message)
            # In case of a rate limit error, wait for 15 seconds and retry
            time.sleep(15)
            if retry_count < 3:
                return self.chat_completion(messages, retry_count=retry_count + 1)
            else:
                logger.error("Max retries reached - unable to complete OpenAI request")
                raise e
        except openai.error.InvalidRequestError as e:
            logger.error(
                f"OpenAI InvalidRequestError. Original error: {e}\nTraceback: {traceback.format_exc()}"
            )
            raise e
        except openai.error.APIConnectionError as e:
            # In case of a api connection error, wait for 60 seconds and retry
            time.sleep(30)
            logger.error(
                f"OpenAI APIConnectionError. Original error: {e}\nTraceback: {traceback.format_exc()}"
            )
            if retry_count < 3:
                return self.chat_completion(messages, retry_count=retry_count + 1)
            else:
                logger.error("Max retries reached - unable to complete OpenAI request")
                raise e
        except Exception as e:
            logger.error(f"Exception: {e}\nTraceback: {traceback.format_exc()}")
            return None
        return response.choices[0].message["content"]

    @staticmethod
    def _extract_json(data_string: str) -> str:
        """
        Extracts a JSON string from a larger string.
        Assumes the JSON content starts with '{' and continues to the end of the input string.
        """
        try:
            start_index = data_string.index("{")
            end_index = data_string.rfind("}")
            json_string = data_string[start_index : end_index + 1]
        except Exception as e:
            print("Failed to extract json", e)
            json_string = data_string
        return json_string

    @staticmethod
    def _load_json_from_text(text):
        """
        Extracts and loads a JSON string from a given text.
        """
        try:
            data = json.loads(text)
        except json.decoder.JSONDecodeError:
            data = None
        return data

    @staticmethod
    def extract_json_from_response(response):
        # In case you cannot handle an error, return None
        if response is None:
            return None
        response_json_format = OpenAiLlmService._extract_json(response)
        response_json = OpenAiLlmService._load_json_from_text(response_json_format)
        return response_json
