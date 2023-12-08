from typing import TypedDict


class OpenAiPromptMessage(TypedDict):
    role: str
    content: str
