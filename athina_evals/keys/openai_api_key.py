from abc import ABC


class OpenAiApiKey(ABC):
    _openai_api_key = None

    @classmethod
    def set_key(cls, api_key):
        cls._openai_api_key = api_key

    @classmethod
    def get_key(cls):
        return cls._openai_api_key
