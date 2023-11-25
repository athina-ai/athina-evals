from abc import ABC


class AthinaApiKey(ABC):
    _athina_api_key = None

    @classmethod
    def set_key(cls, api_key):
        cls._athina_api_key = api_key

    @classmethod
    def get_key(cls):
        return cls._athina_api_key

    @classmethod
    def is_set(cls):
        return cls._athina_api_key is not None
