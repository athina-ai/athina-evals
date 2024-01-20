from abc import ABC, abstractmethod
from typing import List, TypedDict, Tuple

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
        total_sentences = len(sentences_with_evidence)
        unsupported_sentences: List[str] = [] # List of unsupported sentences
        supported_sentences: List[Tuple[str, List[str]]] = [] # List of (sentence, evidences) pairs
        for sentence_with_evidence in sentences_with_evidence:
            sentence_str = sentence_with_evidence.get('sentence')
            supported_evidence_for_sentence = sentence_with_evidence.get('supporting_evidence', [])
            if len(supported_evidence_for_sentence) != 0:
                supported_sentences.append((sentence_str, supported_evidence_for_sentence))
            else:
                unsupported_sentences.append(sentence_str)
        num_supported_sentences = len(supported_sentences)
        score = num_supported_sentences / total_sentences
        precision = 4
        score = round(score, precision)
        return score, unsupported_sentences, supported_sentences
            