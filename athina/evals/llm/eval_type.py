from enum import Enum


class AthinaEvalTypeId(Enum):
    CONTEXT_CONTAINS_ENOUGH_INFORMATION = "ContextContainsEnoughInformation"
    DOES_RESPONSE_ANSWER_QUERY = "DoesResponseAnswerQuery"
    FAITHFULNESS = "Faithfulness"
    CUSTOM = "Custom"
