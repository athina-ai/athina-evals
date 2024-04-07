from enum import Enum

class ConversationEvalTypeId(Enum):
    CONVERSATION_RESOLUTION = "ConversationResolution"
    CONVERSATION_COHERENCE = "ConversationCoherence"

class LlmEvalTypeId(Enum):
    CONTEXT_CONTAINS_ENOUGH_INFORMATION = "Ccei"
    DOES_RESPONSE_ANSWER_QUERY = "Draq"
    FAITHFULNESS = "Irftc"
    GRADING_CRITERIA = "GradingCriteria"
    CUSTOM_PROMPT = "CustomPrompt"
    SUMMARIZATION_HAL = "SummarizationHal"

class RagasEvalTypeId(Enum):
    RAGAS_CONTEXT_RELEVANCY = "RagasContextRelevancy"
    RAGAS_ANSWER_RELEVANCY = "RagasAnswerRelevancy"
    RAGAS_CONTEXT_PRECISION = "RagasContextPrecision"
    RAGAS_FAITHFULNESS = "RagasFaithfulness"
    RAGAS_CONTEXT_RECALL = "RagasContextRecall"
    RAGAS_ANSWER_SEMANTIC_SIMILARITY = "RagasAnswerSemanticSimilarity"
    RAGAS_ANSWER_CORRECTNESS = "RagasAnswerCorrectness"
    RAGAS_HARMFULNESS = "RagasHarmfulness"
    RAGAS_MALICIOUSNESS = "RagasMaliciousness"
    RAGAS_COHERENCE = "RagasCoherence"
    RAGAS_CONCISENESS = "RagasConciseness"


class FunctionEvalTypeId(Enum):
    REGEX = "Regex"
    CONTAINS_ANY = "ContainsAny"
    CONTAINS_ALL = "ContainsAll"
    CONTAINS = "Contains"
    CONTAINS_NONE = "ContainsNone"
    CONTAINS_JSON = "ContainsJson"
    CONTAINS_EMAIL = "ContainsEmail"
    IS_JSON = "IsJson"
    IS_EMAIL = "IsEmail"
    NO_INVALID_LINKS = "NoInvalidLinks"
    CONTAINS_LINK = "ContainsLink"
    CONTAINS_VALID_LINK = "ContainsValidLink"
    EQUALS = "Equals"
    STARTS_WITH = "StartsWith"
    ENDS_WITH = "EndsWith"
    LENGTH_LESS_THAN = "LengthLessThan"
    LENGTH_GREATER_THAN = "LengthGreaterThan"
    API_CALL = "ApiCall"
    SAFE_FOR_WORK_TEXT = "SafeForWorkText"
    NOT_GIBBERISH_TEXT = "NotGibberishText"
    CONTAINS_NO_SENSITIVE_TOPICS = "ContainsNoSensitiveTopics"
    OPENAI_CONTENT_MODERATION = "OpenAiContentModeration"
    
class GroundedEvalTypeId(Enum):
    ANSWER_SIMILARITY = "AnswerSimilarity"
    CONTEXT_SIMILARITY = "ContextSimilarity"

def is_llm_eval(evaluator_type: str) -> bool:
    return any(evaluator_type == member.value for member in LlmEvalTypeId)

def is_ragas_eval(evaluator_type: str) -> bool:
    return any(evaluator_type == member.value for member in RagasEvalTypeId)

def is_function_eval(evaluator_type: str) -> bool:
    return any(evaluator_type == member.value for member in FunctionEvalTypeId)

def is_grounded_eval(evaluator_type: str) -> bool:
    return any(evaluator_type == member.value for member in GroundedEvalTypeId)
