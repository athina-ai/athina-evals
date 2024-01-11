from enum import Enum


class LlmEvalTypeId(Enum):
    CONTEXT_CONTAINS_ENOUGH_INFORMATION = "Ccei"
    DOES_RESPONSE_ANSWER_QUERY = "Draq"
    FAITHFULNESS = "Irftc"
    GRADING_CRITERIA = "GradingCriteria"
    CUSTOM_PROMPT = "CustomPrompt"
    SUMMARY_ACCURACY = "SummaryAccuracy"

class RagasEvalTypeId(Enum):
    RAGAS_CONTEXT_RELEVANCY = "RagasContextRelevancy"
    RAGAS_ANSWER_RELEVANCY = "RagasAnswerRelevancy"

class FunctionEvalTypeId(Enum):
    REGEX = "Regex"
    CONTAINS_ANY = "ContainsAny"