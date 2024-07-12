import os
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from athina.helpers.json import JsonExtractor
from athina.interfaces.model import Model
from athina.steps.base import Step
from athina.llms.abstract_llm_service import AbstractLlmService
from athina.keys import OpenAiApiKey
from athina.llms.openai_service import OpenAiService
from jinja2 import Environment
from athina.helpers.jinja_helper import PreserveUndefined
from athina.steps.transform import ExtractJsonFromString


class PromptMessage(BaseModel):
    role: str
    content: str

class ModelOptions(BaseModel):
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


class PromptTemplate(BaseModel):
    messages: List[PromptMessage]
    env: Environment = None

    class Config:
        arbitrary_types_allowed = True
        
    @staticmethod
    def simple(message: str) -> "PromptTemplate":
        """Create a PromptTemplate from a string representation."""
        messages = [PromptMessage(role="user", content=message)]
        return PromptTemplate(messages=messages)

    def resolve(self, **kwargs) -> List[PromptMessage]:
        """Render the template with given variables."""

        # Create a custom Jinja2 environment with double curly brace delimiters and PreserveUndefined
        self.env = Environment(
            variable_start_string='{{', 
            variable_end_string='}}',
            undefined=PreserveUndefined
        )
        resolved_messages = []
        for message in self.messages:
            content_template = self.env.from_string(message.content)
            content = content_template.render(**kwargs)
            resolved_message = PromptMessage(role=message.role, content=content)
            resolved_messages.append(resolved_message)

        return resolved_messages


class PromptExecution(Step):
    """
    Step that executes a prompt using an LLM service.

    Attributes:
        llm_service (AbstractLlmService): The LLM service to use for prompt execution.
        template (PromptTemplate): The template to render the prompt.
        model (str): The model to use for the LLM service.
    """

    llm_service: AbstractLlmService = None
    template: PromptTemplate
    model: str
    model_options: ModelOptions

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if kwargs.get("llm_service"):
            self.llm_service = kwargs.get("llm_service")
        else:
            self.llm_service = OpenAiService()

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def simple(message: str, model: str = Model.GPT4_O.value) -> "PromptExecution":
        OpenAiApiKey.set_key(os.getenv("OPENAI_API_KEY"))
        openai_service = OpenAiService()
        return PromptExecution(
            llm_service=openai_service,
            template=PromptTemplate.simple(message),
            model=model,
        )

    def execute(self, input_data: dict, **kwargs) -> str:
        """Execute a prompt with the LLM service."""
        if input_data is None:
            input_data = {}

        if not isinstance(input_data, dict) and self.input_key:
            raise ValueError("PromptExecution Error: Input data must be a dictionary")

        response = self.template.resolve(**input_data)
        response = self.llm_service.chat_completion(
            response, model=self.model, **self.model_options.model_dump()
        )

        output_type = kwargs.get('output_type', None)

        if output_type:
            if output_type == "string":
                if not isinstance(response, str):
                    raise ValueError("LLM service response is not a string")
            elif output_type == "array":
                extracted_response = ExtractJsonFromString().execute(response)
                if not isinstance(extracted_response, list):
                    raise ValueError("LLM service response is not an array")
                response = extracted_response

            elif output_type == "object":
                extracted_response = ExtractJsonFromString().execute(response)
                if not isinstance(extracted_response, dict):
                    raise ValueError("LLM service response is not an object")
                response = extracted_response

        elif not isinstance(response, str):
            raise ValueError("LLM service response is not a string")

        return response
