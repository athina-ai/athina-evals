import pkg_resources
import requests
from dataclasses import asdict
from retrying import retry
from typing import List, Optional, Dict
from athina.errors.exceptions import NoAthinaApiKeyException
from athina.interfaces.athina import (
    AthinaFilters,
    AthinaInference,
    AthinaEvalRequestCreateRequest,
    AthinaEvalResultCreateRequest,
    AthinaExperiment,
)
from athina.interfaces.result import EvalPerformanceReport
from athina.keys import AthinaApiKey
from athina.helpers.constants import API_BASE_URL
from athina.errors.exceptions import CustomException

SDK_VERSION = pkg_resources.get_distribution("athina").version


class AthinaApiService:
    @staticmethod
    def _headers():
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
            endpoint = f"{API_BASE_URL}/api/v1/sdk/prompt_run/fetch-by-filter"
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
            if response.status_code == 401:
                response_json = response.json()
                error_message = response_json.get("error", "Unknown Error")
                details_message = "please check your athina api key and try again"
                raise CustomException(error_message, details_message)
            elif response.status_code != 200 and response.status_code != 201:
                response_json = response.json()
                error_message = response_json.get("error", "Unknown Error")
                details_message = response_json.get("details", {}).get(
                    "message", "No Details"
                )
                raise CustomException(error_message, details_message)
            inferences = response.json()["data"]["inferences"]
            return list(map(lambda x: AthinaInference(**x), inferences))
        except Exception as e:
            print("Exception fetching inferences", e)
            pass

    @staticmethod
    def log_usage(eval_name: str, run_type: str):
        """
        Logs a usage event to Posthog via Athina.
        """
        if not AthinaApiKey.is_set():
            return
        try:
            endpoint = f"{API_BASE_URL}/api/v1/sdk/log-usage"
            requests.post(
                endpoint,
                headers=AthinaApiService._headers(),
                json={
                    "sdkVersion": SDK_VERSION,
                    "evalName": eval_name,
                    "run_type": run_type,
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
            if response.status_code == 401:
                response_json = response.json()
                error_message = response_json.get("error", "Unknown Error")
                details_message = "please check your athina api key and try again"
                raise CustomException(error_message, details_message)
            elif response.status_code != 200 and response.status_code != 201:
                response_json = response.json()
                error_message = response_json.get("error", "Unknown Error")
                details_message = response_json.get("details", {}).get(
                    "message", "No Details"
                )
                raise CustomException(error_message, details_message)
            return response.json()
        except Exception as e:
            print(
                f"An error occurred while posting eval results",
                str(e),
            )
            raise

    @staticmethod
    def create_dataset(dataset: Dict):
        """
        Creates a dataset by calling the Athina API
        """
        try:
            endpoint = f"{API_BASE_URL}/api/v1/dataset_v2"
            response = requests.post(
                endpoint,
                headers=AthinaApiService._headers(),
                json=dataset,
            )
            if response.status_code == 401:
                response_json = response.json()
                error_message = response_json.get("error", "Unknown Error")
                details_message = "please check your athina api key and try again"
                raise CustomException(error_message, details_message)
            elif response.status_code != 200 and response.status_code != 201:
                response_json = response.json()
                error_message = response_json.get("error", "Unknown Error")
                details_message = response_json.get("details", {}).get(
                    "message", "No Details"
                )
                raise CustomException(error_message, details_message)
            return response.json()["data"]["dataset"]
        except Exception as e:
            raise

    @staticmethod
    def fetch_dataset_rows(dataset_id: str, number_of_rows: Optional[int] = None):
        """
        Fetch the dataset rows by calling the Athina API

        """
        try:
            if number_of_rows is None:
                number_of_rows = 20
            endpoint = f"{API_BASE_URL}/api/v1/dataset_v2/fetch-by-id/{dataset_id}?offset=0&limit={number_of_rows}&include_dataset_rows=true"
            response = requests.post(endpoint, headers=AthinaApiService._headers())
            if response.status_code == 401:
                response_json = response.json()
                error_message = response_json.get("error", "Unknown Error")
                details_message = "please check your athina api key and try again"
                raise CustomException(error_message, details_message)
            elif response.status_code != 200 and response.status_code != 201:
                response_json = response.json()
                error_message = response_json.get("error", "Unknown Error")
                details_message = response_json.get("details", {}).get(
                    "message", "No Details"
                )
                raise CustomException(error_message, details_message)
            return response.json()["data"]["dataset_rows"]
        except Exception as e:
            raise

    @staticmethod
    def add_dataset_rows(dataset_id: str, rows: List[Dict]):
        """
        Adds rows to a dataset by calling the Athina API.

        Parameters:
        - dataset_id (str): The ID of the dataset to which rows are added.
        - rows (List[Dict]): A list of rows to add to the dataset, where each row is represented as a dictionary.

        Returns:
        The API response data for the dataset after adding the rows.

        Raises:
        - CustomException: If the API call fails or returns an error.
        """
        try:
            endpoint = f"{API_BASE_URL}/api/v1/dataset_v2/{dataset_id}/add-rows"
            response = requests.post(
                endpoint,
                headers=AthinaApiService._headers(),
                json={"dataset_rows": rows},
            )
            if response.status_code == 401:
                response_json = response.json()
                error_message = response_json.get("error", "Unknown Error")
                details_message = "please check your athina api key and try again"
                raise CustomException(error_message, details_message)
            elif response.status_code != 200 and response.status_code != 201:
                response_json = response.json()
                error_message = response_json.get("error", "Unknown Error")
                details_message = response_json.get("details", {}).get(
                    "message", "No Details"
                )
                raise CustomException(error_message, details_message)
            return response.json()["data"]
        except Exception as e:
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
            if response.status_code == 401:
                response_json = response.json()
                error_message = response_json.get("error", "Unknown Error")
                details_message = "please check your athina api key and try again"
                raise CustomException(error_message, details_message)
            elif response.status_code != 200 and response.status_code != 201:
                response_json = response.json()
                error_message = response_json.get("error", "Unknown Error")
                details_message = response_json.get("details", {}).get(
                    "message", "No Details"
                )
                raise CustomException(error_message, details_message)
            return response.json()
        except Exception as e:
            print(
                f"An error occurred while creating eval request",
                str(e),
            )
            raise

    def log_eval_performance_report(
        self, eval_request_id: str, report: EvalPerformanceReport
    ):
        """
        Logs the performance metrics for the evaluator.
        """
        try:
            endpoint = f"{API_BASE_URL}/api/v1/eval_performance_report"
            response = requests.post(
                endpoint,
                headers=AthinaApiService._headers(),
                json={
                    "eval_request_id": eval_request_id,
                    "true_positives": report["true_positives"],
                    "false_positives": report["false_positives"],
                    "true_negatives": report["true_negatives"],
                    "false_negatives": report["false_negatives"],
                    "accuracy": report["accuracy"],
                    "precision": report["precision"],
                    "recall": report["recall"],
                    "f1_score": report["f1_score"],
                    "runtime": report["runtime"],
                    "dataset_size": report["dataset_size"],
                },
            )
            if response.status_code == 401:
                response_json = response.json()
                error_message = response_json.get("error", "Unknown Error")
                details_message = "please check your athina api key and try again"
                raise CustomException(error_message, details_message)
            elif response.status_code != 200 and response.status_code != 201:
                response_json = response.json()
                error_message = response_json.get("error", "Unknown Error")
                details_message = response_json.get("details", {}).get(
                    "message", "No Details"
                )
                raise CustomException(error_message, details_message)
            return response.json()
        except Exception as e:
            print(
                f"An error occurred while posting eval performance report",
                str(e),
            )
            raise

    @staticmethod
    def log_experiment(
        eval_request_id: str,
        experiment: AthinaExperiment,
    ):
        """
        Logs the experiment metadata to Athina.
        """
        try:
            endpoint = f"{API_BASE_URL}/api/v1/experiment"
            response = requests.post(
                endpoint,
                headers=AthinaApiService._headers(),
                json={
                    "eval_request_id": eval_request_id,
                    "experiment_name": experiment["experiment_name"],
                    "experiment_description": experiment["experiment_description"],
                    "language_model_provider": experiment["language_model_provider"],
                    "language_model_id": experiment["language_model_id"],
                    "prompt_template": experiment["prompt_template"],
                    "dataset_name": experiment["dataset_name"],
                },
            )
            print(response.status_code)
            if response.status_code == 401:
                response_json = response.json()
                error_message = response_json.get("error", "Unknown Error")
                details_message = "please check your athina api key and try again"
                raise CustomException(error_message, details_message)
            elif response.status_code != 200 and response.status_code != 201:
                response_json = response.json()
                error_message = response_json.get("error", "Unknown Error")
                details_message = response_json.get("details", {}).get(
                    "message", "No Details"
                )
                raise CustomException(error_message, details_message)
            return response.json()
        except Exception as e:
            print(
                f"An error occurred while posting experiment metadata",
                str(e),
            )
            raise

    @staticmethod
    def log_eval_results_with_config(eval_results_with_config: dict):
        try:
            endpoint = f"{API_BASE_URL}/api/v1/eval_run/log-eval-results-sdk"
            response = requests.post(
                endpoint,
                headers=AthinaApiService._headers(),
                json=eval_results_with_config,
            )
            if response.status_code == 401:
                response_json = response.json()
                error_message = response_json.get("error", "Unknown Error")
                details_message = "please check your athina api key and try again"
                raise CustomException(error_message, details_message)
            elif response.status_code != 200 and response.status_code != 201:
                response_json = response.json()
                error_message = response_json.get("error", "Unknown Error")
                details_message = response_json.get("details", {}).get(
                    "message", "No Details"
                )
                raise CustomException(error_message, details_message)
            return response.json()
        except Exception as e:
            raise
