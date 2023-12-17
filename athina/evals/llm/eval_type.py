from enum import Enum


class AthinaEvalTypeId(Enum):
    CONTEXT_CONTAINS_ENOUGH_INFORMATION = "Ccei"
    DOES_RESPONSE_ANSWER_QUERY = "Draq"
    FAITHFULNESS = "Irftc"
    CUSTOM = "Custom"
    SUMMARY_ACCURACY = "SummaryAccuracy"
