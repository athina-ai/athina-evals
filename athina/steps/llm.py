import os
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from athina.helpers.json import JsonExtractor
from athina.interfaces.model import Model
from athina.steps.base import Step
from athina.llms.abstract_llm_service import AbstractLlmService
from athina.keys import OpenAiApiKey
from athina.llms.openai_service import OpenAiService


class PromptMessage(BaseModel):
    role: str
    content: str


class PromptTemplate(BaseModel):
    messages: List[PromptMessage]

    @staticmethod
    def simple(message: str) -> "PromptTemplate":
        """Create a PromptTemplate from a string representation."""
        messages = [PromptMessage(role="user", content=message)]
        return PromptTemplate(messages=messages)

    def resolve(self, **kwargs) -> List[PromptMessage]:
        """Render the template with given variables."""
        resolved_messages = []
        for message in self.messages:
            content = message.content.format(**kwargs)
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

    llm_service: AbstractLlmService
    template: PromptTemplate
    model: str

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

    def execute(self, input_data: dict) -> str:
        """Execute a prompt with the LLM service."""
        if input_data is None:
            input_data = {}

        if (input_data is not isinstance(input_data, dict)) and self.input_key:
            raise ValueError("PromptExecution Error: Input data must be a dictionary")

        response = self.template.resolve(**input_data)
        response = self.llm_service.chat_completion(
            response, model=self.model
        )  # Simulated LLM service call

        if type(response) is not str:
            raise ValueError("LLM service response is not a string")

        return response
