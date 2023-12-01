import pkg_resources
import requests
from retrying import retry
from typing import List, Optional, Any
from athina.interfaces.athina import (
    AthinaFilters,
    AthinaInference,
    AthinaEvalRequestCreateRequest,
    AthinaEvalResultCreateRequest,
)
from athina.keys import AthinaApiKey
from athina.helpers.constants import API_BASE_URL

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
        filters: Optional[AthinaFilters], limit: int
    ) -> List[AthinaInference]:
        """
        Load data from Athina API.
        """
        try:
            endpoint = f"{API_BASE_URL}/api/v1/sdk/prompt-runs/fetch-by-filter"
            filters_dict = filters.to_dict() if filters is not None else {}
            json = {
                "limit": limit,
                **filters_dict,
            }
            json = {k: v for k, v in json.items() if v is not None}
            response = requests.post(
                endpoint,
                headers=AthinaApiService._headers(),
                json=json,
            )
            inferences = response.json()["data"]["inferences"]
            return list(map(lambda x: AthinaInference(**x), inferences))
        except Exception as e:
            print("Exception fetching inferences", e)
            pass

    @staticmethod
    def log_usage(eval_name: str) -> List[AthinaInference]:
        """
        Logs a usage event to Posthog via Athina.
        """
        try:
            endpoint = f"{API_BASE_URL}/api/v1/sdk/log-usage"
            requests.post(
                endpoint,
                headers=AthinaApiService._headers(),
                json={
                    "sdkVersion": SDK_VERSION,
                    "evalName": eval_name,
                },
            )
        except Exception as e:
            # Silent failure is ok here.
            pass

    @staticmethod
    @retry(wait_fixed=500, stop_max_attempt_number=3)
    def log_eval_results(
        athina_eval_result_create_many_request: List[AthinaEvalResultCreateRequest],
    ):
        """
        Logs eval results to Athina
        """
        try:
            # Construct eval update requests
            endpoint = f"{API_BASE_URL}/api/v1/eval_result"
            response = requests.post(
                endpoint,
                headers=AthinaApiService._headers(),
                json=athina_eval_result_create_many_request,
            )
            return response.json()
        except Exception as e:
            print(
                f"An error occurred while posting eval results",
                str(e),
            )
            raise

    @staticmethod
    def create_eval_request(
        athina_eval_request_create_request: AthinaEvalRequestCreateRequest,
    ):
        """
        Create eval request
        """
        try:
            endpoint = f"{API_BASE_URL}/api/v1/eval_request"
            response = requests.post(
                endpoint,
                headers=AthinaApiService._headers(),
                json=athina_eval_request_create_request,
            )
            return response.json()
        except Exception as e:
            print(
                f"An error occurred while posting eval results",
                str(e),
            )
            raise
