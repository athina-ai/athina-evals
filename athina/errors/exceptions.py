from typing import Optional
from .messages import NO_ATHINA_API_KEY, NO_OPENAI_API_KEY


class CustomException(Exception):
    def __init__(
        self, message: Optional[str] = None, extra_info: Optional[dict] = None
    ):
        self.message = message
        self.extra_info = extra_info
        super().__init__(self.message)

    def __str__(self):
        if self.extra_info:
            return f"{self.message} (Extra Info: {self.extra_info})"
        return self.message


class NoAthinaApiKeyException(CustomException):
    def __init__(self, message: str = NO_ATHINA_API_KEY):
        super().__init__(message)


class NoOpenAiApiKeyException(CustomException):
    def __init__(self, message: str = NO_OPENAI_API_KEY):
        super().__init__(message)
