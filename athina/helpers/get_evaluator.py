from athina.evals import Regex, ContainsAny, ContainsAll, Contains, ContainsNone, ContainsJson, ContainsEmail, IsJson, IsEmail, NoInvalidLinks, ContainsLink, ContainsValidLink, Equals, StartsWith, EndsWith, LengthLessThan, LengthGreaterThan, ApiCall, DoesResponseAnswerQuery, Faithfulness, BaseEvaluator, ContextContainsEnoughInformation, SummaryAccuracy, Groundedness, GradingCriteria, CustomPrompt, RagasContextRelevancy, RagasAnswerRelevancy, RagasAnswerCorrectness, RagasAnswerSemanticSimilarity, RagasCoherence, RagasConciseness, RagasContextPrecision, RagasContextRecall, RagasFaithfulness, RagasHarmfulness, RagasMaliciousness

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
    "ApiCall": ApiCall,
}

llm_operations = {
    "Draq": DoesResponseAnswerQuery,
    "Irftc": Faithfulness,
    "BaseEvaluator": BaseEvaluator,
    "Ccei": ContextContainsEnoughInformation,
    "SummaryAccuracy": SummaryAccuracy,
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
    "RagasMaliciousness": RagasMaliciousness
}


def get_evaluator(evaluator_type):
    if evaluator_type in function_operations:
        return function_operations[evaluator_type]
    elif evaluator_type in llm_operations:
        return llm_operations[evaluator_type]
    elif evaluator_type in ragas_operations:
        return ragas_operations[evaluator_type]
    else:
        raise ValueError(f"Invalid evaluator type: {evaluator_type}")
    

