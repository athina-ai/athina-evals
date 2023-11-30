import asyncio
import pkg_resources
import requests
from typing import List, Optional
from athina.interfaces.athina import AthinaFilters, AthinaInference
from athina.keys import AthinaApiKey

BASE_API_URL = "https://log.athina.ai"

SDK_VERSION = pkg_resources.get_distribution("athina-evals").version


class AthinaApiService:
    @staticmethod
    def _headers():
        if not AthinaApiKey.is_set():
            raise Exception(
                """Please sign up at https://athina.ai and set an Athina API key. 
                See https://docs.athina.ai/evals/quick_start for more information."""
            )
        athina_api_key = AthinaApiKey.get_key()
        return {
            "athina-api-key": athina_api_key,
        }

    @staticmethod
    def fetch_inferences(
        filters: Optional[AthinaFilters], limit: int = 50
    ) -> List[AthinaInference]:
        """
        Load data from Athina API.
        """
        raise NotImplementedError("This method has not been implemented yet.")

    @staticmethod
    def log_eval_results() -> List[AthinaInference]:
        """
        Logs eval results to Athina
        """
        pass

    @staticmethod
    def log_usage(evalName: str) -> List[AthinaInference]:
        """
        Logs a usage event to Posthog via Athina.
        """
        try:
            endpoint = f"{BASE_API_URL}/api/v1/sdk/log-usage"
            requests.post(
                endpoint,
                headers=AthinaApiService._headers(),
                json={
                    "sdkVersion": SDK_VERSION,
                    "evalName": evalName,
                },
            )
        except Exception as e:
            # Silent failure is ok here.
            pass
