from enum import Enum


class Model(Enum):
    """
    Supported models for evaluations.
    """

    GPT35_TURBO = "gpt-3.5-turbo"
    GPT4 = "gpt-4"
    GPT4_1106_PREVIEW = "gpt-4-1106-preview"

    @staticmethod
    def is_supported(model_name: str) -> bool:
        """
        Checks if the model is supported.
        """
        return model_name in [model.value for model in Model]

    @staticmethod
    def supports_json_mode(model_name: str) -> bool:
        """
        Checks if the model supports json mode.
        """
        JSON_MODE_SUPPORTED_MODELS = [Model.GPT4_1106_PREVIEW]
        return model_name in [model.value for model in JSON_MODE_SUPPORTED_MODELS]
