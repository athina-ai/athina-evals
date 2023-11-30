import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List


@dataclass
class AthinaInference:
    """Athina PromptRun class"""

    id: str
    prompt_slug: Optional[str]
    language_model_id: Optional[str]
    user_query: Optional[str]
    context: Optional[Dict[str, str]]
    prompt_response: Optional[str]


@dataclass
class AthinaFilters:
    prompt_slug: Optional[str] = None
    language_model_id: Optional[str] = None
    environment: Optional[str] = None
    topic: Optional[str] = None
    customer_id: Optional[str] = None

    def to_dict(self) -> str:
        return asdict(self)
