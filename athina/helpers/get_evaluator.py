from athina.evals import (
    Regex,
    ContainsAny,
    ContainsAll,
    Contains,
    ContainsNone,
    ContainsJson,
    ContainsEmail,
    IsJson,
    IsEmail,
    NoInvalidLinks,
    ContainsLink,
    ContainsValidLink,
    Equals,
    StartsWith,
    EndsWith,
    LengthLessThan,
    LengthGreaterThan,
    LengthBetween,
    ApiCall,
    DoesResponseAnswerQuery,
    Faithfulness,
    BaseEvaluator,
    ContextContainsEnoughInformation,
    SummaryAccuracy,
    Groundedness,
    GradingCriteria,
    CustomPrompt,
    RagasContextRelevancy,
    RagasAnswerRelevancy,
    RagasAnswerCorrectness,
    RagasAnswerSemanticSimilarity,
    RagasCoherence,
    RagasConciseness,
    RagasContextPrecision,
    RagasContextRecall,
    RagasFaithfulness,
    RagasHarmfulness,
    RagasMaliciousness,
    NotGibberishText,
    SafeForWorkText,
    ContainsNoSensitiveTopics,
    OpenAiContentModeration,
    PiiDetection,
    PromptInjection,
    ProfanityFree,
    ReadingTime,
    DetectPII,
    ToxicLanguage,
    CorrectLanguage,
    NoSecretsPresent,
    RestrictToTopic,
    NotUnusualPrompt,
    PolitenessCheck,
    OneLine,
    JsonSchema,
    JsonValidation,
    CustomCodeEval,
    ConversationResolution,
    ConversationCoherence,
)
from athina.evals.grounded.similarity import (
    CosineSimilarity,
    JaccardSimilarity,
    JaroWincklerSimilarity,
    NormalisedLevenshteinSimilarity,
    SorensenDiceSimilarity,
)
from athina.evals.grounded.wrapper import AnswerSimilarity, ContextSimilarity

grounded_operations = {
    "AnswerSimilarity": AnswerSimilarity,
    "ContextSimilarity": ContextSimilarity,
}

conversation_operations = {
    "ConversationResolution": ConversationResolution,
    "ConversationCoherence": ConversationCoherence,
}

function_operations = {
    "Regex": Regex,
    "ContainsAny": ContainsAny,
    "ContainsAll": ContainsAll,
    "Contains": Contains,
    "ContainsNone": ContainsNone,
    "ContainsJson": ContainsJson,
    "ContainsEmail": ContainsEmail,
    "IsJson": IsJson,
    "IsEmail": IsEmail,
    "NoInvalidLinks": NoInvalidLinks,
    "ContainsLink": ContainsLink,
    "ContainsValidLink": ContainsValidLink,
    "Equals": Equals,
    "StartsWith": StartsWith,
    "EndsWith": EndsWith,
    "LengthLessThan": LengthLessThan,
    "LengthGreaterThan": LengthGreaterThan,
    "LengthBetween": LengthBetween,
    "ApiCall": ApiCall,
    "OneLine": OneLine,
    "JsonSchema": JsonSchema,
    "JsonValidation": JsonValidation,
    "CustomCodeEval": CustomCodeEval,
}

safety_operations = {
    "SafeForWorkText": SafeForWorkText,
    "NotGibberishText": NotGibberishText,
    "ContainsNoSensitiveTopics": ContainsNoSensitiveTopics,
    "OpenAiContentModeration": OpenAiContentModeration,
    "PiiDetection": PiiDetection,
    "PromptInjection": PromptInjection,
    "ProfanityFree": ProfanityFree,
    "ReadingTime": ReadingTime,
    "DetectPII": DetectPII,
    "ToxicLanguage": ToxicLanguage,
    "CorrectLanguage": CorrectLanguage,
    "NoSecretsPresent": NoSecretsPresent,
    "RestrictToTopic": RestrictToTopic,
    "NotUnusualPrompt": NotUnusualPrompt,
    "PolitenessCheck": PolitenessCheck,
}

llm_operations = {
    "Draq": DoesResponseAnswerQuery,
    "Irftc": Faithfulness,
    "BaseEvaluator": BaseEvaluator,
    "Ccei": ContextContainsEnoughInformation,
    "SummarizationHal": SummaryAccuracy,
    "Groundedness": Groundedness,
    "GradingCriteria": GradingCriteria,
    "CustomPrompt": CustomPrompt,
}

ragas_operations = {
    "RagasContextRelevancy": RagasContextRelevancy,
    "RagasAnswerRelevancy": RagasAnswerRelevancy,
    "RagasAnswerCorrectness": RagasAnswerCorrectness,
    "RagasAnswerSemanticSimilarity": RagasAnswerSemanticSimilarity,
    "RagasCoherence": RagasCoherence,
    "RagasConciseness": RagasConciseness,
    "RagasContextPrecision": RagasContextPrecision,
    "RagasContextRecall": RagasContextRecall,
    "RagasFaithfulness": RagasFaithfulness,
    "RagasHarmfulness": RagasHarmfulness,
    "RagasMaliciousness": RagasMaliciousness,
}


def get_evaluator(evaluator_type):
    if evaluator_type in function_operations:
        return function_operations[evaluator_type]
    elif evaluator_type in safety_operations:
        return safety_operations[evaluator_type]
    elif evaluator_type in grounded_operations:
        return grounded_operations[evaluator_type]
    elif evaluator_type in llm_operations:
        return llm_operations[evaluator_type]
    elif evaluator_type in ragas_operations:
        return ragas_operations[evaluator_type]
    elif evaluator_type in conversation_operations:
        return conversation_operations[evaluator_type]
    else:
        raise ValueError(f"Invalid evaluator type: {evaluator_type}")


# TODO : Remove the following methods from workers repo to reduce code duplication
def get_comparator(comparator_name):
    if comparator_name is None:
        raise ValueError("similarity_function is a required argument")
    comparators = {
        "CosineSimilarity": CosineSimilarity(),
        "NormalisedLevenshteinSimilarity": NormalisedLevenshteinSimilarity(),
        "JaroWincklerSimilarity": JaroWincklerSimilarity(),
        "JaccardSimilarity": JaccardSimilarity(),
        "SorensenDiceSimilarity": SorensenDiceSimilarity(),
    }
    comparator = comparators.get(comparator_name, None)
    if comparator is None:
        raise NotImplementedError(f"Comparator {comparator_name} not implemented.")
    return comparator


def create_grounded_evaluator(grounded_eval_name, comparator, failure_threshold):
    grounded_evaluator_class = grounded_operations.get(grounded_eval_name, None)
    if grounded_evaluator_class is None:
        raise NotImplementedError(
            f"Grounded eval {grounded_eval_name} not implemented."
        )
    else:
        return grounded_evaluator_class(
            comparator=comparator, failure_threshold=failure_threshold
        )
