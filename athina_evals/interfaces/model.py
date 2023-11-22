from enum import Enum


class Model(Enum):
    """
    Supported models for evaluations.

    Args:
        Enum (_type_): _description_
    """

    GPT35_TURBO = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    GPT4_1106_PREVIEW = "gpt-4-1106-preview"
