from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class AthinaInference:
    """Athina PromptRun class"""

    prompt_response: Optional[str]
    prompt_slug: Optional[str]
    language_model_id: Optional[str]
    environment: Optional[str]
    topic: Optional[str]
    customer_id: Optional[str]
    context: Optional[dict]
    user_query: Optional[str]


@dataclass
class AthinaEvalResult:
    """Athina PromptRun class"""

    # TODO: Implement this
    pass


@dataclass
class AthinaFilters:
    prompt_slug: Optional[str]
    language_model_id: Optional[str]
    environment: Optional[str]
    topic: Optional[str]
    customer_id: Optional[str]
