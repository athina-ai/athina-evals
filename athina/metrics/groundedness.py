from abc import ABC, abstractmethod
from typing import List, Dict, TypedDict

class GroundednessEvidence(TypedDict):
    sentence: str
    supporting_evidence: List[str]

class GroundednessScore(ABC):
    """
    Computes the groundedness score.
    """

    @staticmethod
    def compute(sentences_with_evidence: List[GroundednessEvidence]):
        """
        Computes the metric.
        """
        supported_sentences = 0
        total_sentences = len(sentences_with_evidence)
        unsupported_sentences: List[str] = []
        for sentence in sentences_with_evidence:
            supported_evidence_for_sentence = sentence.get('supporting_evidence', [])
            if len(supported_evidence_for_sentence) != 0:
                supported_sentences += 1
                unsupported_sentences.append(sentence.get('sentence'))
        score = supported_sentences / total_sentences
        precision = 4
        score = round(score, precision)
        return score, unsupported_sentences
            