from enum import Enum


class Model(Enum):
    """
    Supported models for evaluations.
    """

    GPT35_TURBO = "gpt-3.5-turbo"
    GPT35_TURBO_1106 = "gpt-3.5-turbo-1106"
    GPT4 = "gpt-4"
    GPT4_1106_PREVIEW = "gpt-4-1106-preview"
    GPT4_TURBO_PREVIEW = "gpt-4-turbo-preview"
    GPT35_TURBO_16K = "gpt-3.5-turbo-16k"
    COMMAND_LIGHT = "command-light"
    COMMAND = "command"
    AZURE_GPT35_TURBO = "azure/gpt-3.5-turbo"
    AZURE_GPT35_TURBO_1106 = "azure/gpt-3.5-turbo-1106"
    AZURE_GPT4 = "azure/gpt-4"
    AZURE_GPT4_1106_PREVIEW = "azure/gpt-4-1106-preview"
    GEMINI_PROD = "gemini/gemini-prod"
    CLAUDE_2 = "claude-2"
    MISTRAL_TINY = "mistral/mistral-tiny"
    MISTRAL_SMALL = "mistral/mistral-small"

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
        JSON_MODE_SUPPORTED_MODELS = [Model.GPT4_1106_PREVIEW, Model.GPT35_TURBO_1106]
        return model_name in [model.value for model in JSON_MODE_SUPPORTED_MODELS]
