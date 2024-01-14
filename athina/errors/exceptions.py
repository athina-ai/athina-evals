from typing import Optional
from athina.constants.messages import AthinaMessages


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
    def __init__(self, message: str = AthinaMessages.SIGN_UP_FOR_BEST_EXPERIENCE):
        super().__init__(message)


class NoOpenAiApiKeyException(CustomException):
    def __init__(self, message: str = AthinaMessages.NO_OPENAI_API_KEY):
        super().__init__(message)
