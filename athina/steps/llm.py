import os
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
from athina.helpers.json import JsonExtractor
from athina.interfaces.model import Model
from athina.steps.base import Step
from athina.llms.abstract_llm_service import AbstractLlmService
from athina.keys import OpenAiApiKey
from athina.llms.openai_service import OpenAiService
from jinja2 import Environment
from athina.helpers.jinja_helper import PreserveUndefined
from athina.steps.transform import ExtractJsonFromString, ExtractNumberFromString
import traceback
import json


class TextContent(BaseModel):
    type: str
    text: str


class ImageContent(BaseModel):
    type: str = "image"
    image_url: Union[str, Dict[str, str]]

    def to_api_format(self):
        if isinstance(self.image_url, dict):
            return {"type": "image_url", "image_url": self.image_url}
        return {"type": "image_url", "image_url": {"url": self.image_url}}


Content = Union[str, List[Union[TextContent, ImageContent]]]


class PromptMessage(BaseModel):
    role: str
    content: Optional[Content] = None
    tool_call: Optional[str] = None

    def to_api_format(self) -> dict:
        """Convert the message to the format expected by the OpenAI API"""
        if self.content is None:
            return {"role": self.role}

        if isinstance(self.content, str):
            return {"role": self.role, "content": self.content}

        if isinstance(self.content, list):
            formatted_content = []
            for item in self.content:
                if isinstance(item, TextContent):
                    formatted_content.append({"type": "text", "text": item.text})
                elif isinstance(item, ImageContent):
                    formatted_content.append(item.to_api_format())
            return {"role": self.role, "content": formatted_content}


class ModelOptions(BaseModel):
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None


class ToolConfig(BaseModel):
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    tools: Optional[List[Any]] = None


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
            variable_start_string="{{",
            variable_end_string="}}",
            undefined=PreserveUndefined,
        )

        final_messages = []
        for message in self.messages:
            if message.role == "import":
                # Find the value wrapped in {{}}
                import_key = message.content.strip("{}")

                # Find the value in the row
                if import_key in kwargs:
                    value = kwargs[import_key]

                    # Check if it is a list/array
                    if isinstance(value, list):
                        # Iterate over the list and create a new PromptMessage for each item
                        for item in value:
                            if isinstance(item, dict):
                                # If item has tool_call, then parse tool_call and create a new PromptMessage
                                if "tool_call" in item:
                                    try:
                                        tool_call_message = PromptMessage(
                                            role=item["role"],
                                            tool_call=self.env.from_string(
                                                item.get("tool_call")
                                            ).render(**kwargs),
                                        )
                                        final_messages.append(tool_call_message)
                                    except Exception as e:
                                        print(f"Error parsing tool_call: {e}")
                                else:
                                    new_message = PromptMessage(**item)
                                    final_messages.append(new_message)
            else:
                final_messages.append(message)

        resolved_messages = []
        for message in final_messages:
            if message.content is None:
                resolved_messages.append(message)
            elif isinstance(message.content, str):
                content_template = self.env.from_string(message.content)
                content = content_template.render(**kwargs)
                resolved_message = PromptMessage(role=message.role, content=content)
                resolved_messages.append(resolved_message)
            elif isinstance(message.content, list):
                resolved_content = []
                for item in message.content:
                    if isinstance(item, TextContent):
                        content_template = self.env.from_string(item.text)
                        resolved_text = content_template.render(**kwargs)
                        resolved_content.append(
                            TextContent(text=resolved_text, type="text")
                        )
                    elif isinstance(item, ImageContent):
                        resolved_content.append(item)
                resolved_message = PromptMessage(
                    role=message.role, content=resolved_content
                )
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
    template: Union[PromptTemplate, dict[str, List[Dict[str, Any]]]]
    model: str
    model_options: ModelOptions
    tool_config: Optional[ToolConfig] = None
    response_format: Optional[dict] = None
    name: Optional[str] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if kwargs.get("llm_service"):
            self.llm_service = kwargs.get("llm_service")
        else:
            self.llm_service = OpenAiService()

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def simple(
        message: str,
        model: str = Model.GPT4_O.value,
        name: Optional[str] = None,
    ) -> "PromptExecution":
        OpenAiApiKey.set_key(os.getenv("OPENAI_API_KEY"))
        openai_service = OpenAiService()
        return PromptExecution(
            llm_service=openai_service,
            template=PromptTemplate.simple(message),
            model=model,
            model_options=ModelOptions(),
        )

    def execute(self, input_data: dict, **kwargs) -> str:
        """Execute a prompt with the LLM service."""
        if input_data is None:
            input_data = {}

        if not isinstance(input_data, dict) and self.input_key:
            raise ValueError("PromptExecution Error: Input data must be a dictionary")

        try:
            messages = self.template.resolve(**input_data)
            # Convert messages to API format
            api_formatted_messages = [msg.to_api_format() for msg in messages]

            llm_service_response = self.llm_service.chat_completion(
                messages,
                model=self.model,
                **self.model_options.model_dump(),
                **(self.tool_config.model_dump() if self.tool_config else {}),
                **({"response_format": self.response_format}),
                **(
                    kwargs.get("search_domain_filter", {})
                    if isinstance(kwargs.get("search_domain_filter"), dict)
                    else {}
                ),
            )
            llmresponse = llm_service_response["value"]
            output_type = kwargs.get("output_type", None)
            error = None
            if output_type:
                if output_type == "string":
                    if not isinstance(llmresponse, str):
                        error = "LLM response is not a string"
                    response = llmresponse

                elif output_type == "number":
                    extracted_response = ExtractNumberFromString().execute(llmresponse)
                    if not isinstance(extracted_response, (int, float)):
                        error = "LLM response is not a number"
                    response = extracted_response

                elif output_type == "array":
                    extracted_response = ExtractJsonFromString().execute(llmresponse)
                    if not isinstance(extracted_response, list):
                        error = "LLM response is not an array"
                    response = extracted_response

                elif output_type == "object":
                    extracted_response = ExtractJsonFromString().execute(llmresponse)
                    if not isinstance(extracted_response, dict):
                        error = "LLM response is not an object"
                    response = extracted_response

            elif not isinstance(llmresponse, str):
                error = "LLM service response is not a string"

            else:
                response = llmresponse

            if error:
                return {"status": "error", "data": error}
            else:
                return {
                    "status": "success",
                    "data": response,
                    "metadata": (
                        json.loads(llm_service_response.get("metadata", "{}"))
                        if llm_service_response.get("metadata")
                        else {}
                    ),
                }
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "data": str(e)}

    async def execute_async(self, input_data: dict, **kwargs) -> dict:
        """Execute a prompt with the LLM service asynchronously."""
        if input_data is None:
            input_data = {}

        if not isinstance(input_data, dict) and self.input_key:
            raise ValueError("PromptExecution Error: Input data must be a dictionary")

        try:
            messages = self.template.resolve(**input_data)
            # Convert messages to API format
            # TODO: Why is api_formatted_messages not used?
            api_formatted_messages = [msg.to_api_format() for msg in messages]

            llm_service_response = await self.llm_service.chat_completion_async(
                messages,
                model=self.model,
                **self.model_options.model_dump(),
                **(self.tool_config.model_dump() if self.tool_config else {}),
                **({"response_format": self.response_format}),
                **(
                    kwargs.get("search_domain_filter", {})
                    if isinstance(kwargs.get("search_domain_filter"), dict)
                    else {}
                ),
            )
            llmresponse = llm_service_response["value"]
            output_type = kwargs.get("output_type", None)
            error = None
            if output_type:
                if output_type == "string":
                    if not isinstance(llmresponse, str):
                        error = "LLM response is not a string"
                    response = llmresponse

                elif output_type == "number":
                    extracted_response = ExtractNumberFromString().execute(llmresponse)
                    if not isinstance(extracted_response, (int, float)):
                        error = "LLM response is not a number"
                    response = extracted_response

                elif output_type == "array":
                    extracted_response = ExtractJsonFromString().execute(llmresponse)
                    if not isinstance(extracted_response, list):
                        error = "LLM response is not an array"
                    response = extracted_response

                elif output_type == "object":
                    extracted_response = ExtractJsonFromString().execute(llmresponse)
                    if not isinstance(extracted_response, dict):
                        error = "LLM response is not an object"
                    response = extracted_response

            elif not isinstance(llmresponse, str):
                error = "LLM service response is not a string"

            else:
                response = llmresponse

            if error:
                return {"status": "error", "data": error}
            else:
                return {
                    "status": "success",
                    "data": response,
                    "metadata": (
                        json.loads(llm_service_response.get("metadata", "{}"))
                        if llm_service_response.get("metadata")
                        else {}
                    ),
                }
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "data": str(e)}
