from enum import Enum


class LlmEvalTypeId(Enum):
    CONTEXT_CONTAINS_ENOUGH_INFORMATION = "Ccei"
    DOES_RESPONSE_ANSWER_QUERY = "Draq"
    FAITHFULNESS = "Irftc"
    CUSTOM = "Custom"

class RagasEvalTypeId(Enum):
    CONTEXT_RELEVANCY = "ContextRelevancy"