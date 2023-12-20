from abc import ABC, abstractmethod
from typing import List, TypedDict, Optional

class QuestionAnswererResponse(TypedDict):
    answer: str
    explanation: Optional[str]

class QuestionAnswerer(ABC):
    
        @abstractmethod
        def answer(self, questions: List[str], context: str) -> QuestionAnswererResponse:
            pass