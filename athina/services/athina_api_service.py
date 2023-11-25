from typing import List, Optional
from athina.interfaces.athina import AthinaFilters, AthinaInference


class AthinaApiService:
    @staticmethod
    def fetch_inferences(
        filters: Optional[AthinaFilters], limit: int = 50
    ) -> List[AthinaInference]:
        """
        Load data from Athina API.
        """
        return [
            AthinaInference(
                prompt_response="abc",
                prompt_slug="slugzy",
                language_model_id="gpt-4",
                environment="prod",
                topic="refunds",
                customer_id=None,
                context={"information": "random info"},
                user_query="who is jon snow?",
            )
        ]
        pass

    @staticmethod
    def log_eval_results() -> List[AthinaInference]:
        """
        Logs eval results to Athina
        """
        pass

    @staticmethod
    def log_usage() -> List[AthinaInference]:
        """
        Logs a usage event to Posthog via Athina.
        """
        pass
