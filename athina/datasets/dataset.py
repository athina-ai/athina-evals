from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from athina.services.athina_api_service import AthinaApiService


@dataclass
class Dataset:
    id: str
    source: str
    name: str
    description: Optional[str] = None
    language_model_id: Optional[str] = None
    prompt_template: Optional[Any] = None
    rows: List[Dict[str, Any]] = field(default_factory=list)

    @staticmethod
    def create(
        name: str,
        description: Optional[str] = None,
        language_model_id: Optional[str] = None,
        prompt_template: Optional[Any] = None,
        rows: List[Dict[str, Any]] = None,
    ):
        dataset_data = {
            "source": "dev_sdk",
            "name": name,
            "description": description,
            "language_model_id": language_model_id,
            "prompt_template": prompt_template,
            "dataset_rows": rows or [],
        }

        dataset_data = {k: v for k, v in dataset_data.items() if v is not None}

        try:
            created_dataset_data = AthinaApiService.create_dataset(dataset_data)
        except Exception as e:
            raise
        dataset = Dataset(
            id=created_dataset_data["id"],
            source=created_dataset_data["source"],
            name=created_dataset_data["name"],
            description=created_dataset_data["description"],
            language_model_id=created_dataset_data["language_model_id"],
            prompt_template=created_dataset_data["prompt_template"],
        )
        return dataset

    @staticmethod
    def add_rows(dataset_id: str, rows: List[Dict[str, Any]]):
        batch_size = 100
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            try:
                AthinaApiService.add_dataset_rows(dataset_id, batch)
            except Exception as e:
                raise

    @staticmethod
    def list_datasets():
        try:
            datasets = AthinaApiService.list_datasets()
        except Exception as e:
            raise
        return [
            Dataset(
                id=dataset["id"],
                source=dataset["source"],
                name=dataset["name"],
                description=dataset["description"],
                language_model_id=dataset["language_model_id"],
                prompt_template=dataset["prompt_template"],
            )
            for dataset in datasets
        ]

    @staticmethod
    def delete_dataset_by_id(dataset_id: str):
        try:
            response = AthinaApiService.delete_dataset_by_id(dataset_id)
            return response
        except Exception as e:
            raise

    @staticmethod
    def get_dataset_by_id(dataset_id: str):
        try:
            response = AthinaApiService.get_dataset_by_id(dataset_id)
            return Dataset._clean_response(response)
        except Exception as e:
            raise

    @staticmethod
    def get_dataset_by_name(name: str):
        try:
            response = AthinaApiService.get_dataset_by_name(name)
            return Dataset._clean_response(response)
        except Exception as e:
            raise

    @staticmethod
    def dataset_link(dataset_id: str):
        return f"https://app.athina.ai/develop/{dataset_id}"

    @staticmethod
    def _clean_response(response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cleans the response by removing unnecessary keys and modifying the dataset_eval_results.
        """
        dataset = response.get("dataset", {})
        dataset_rows = response.get("dataset_rows", [])
        development_eval_configs = response.get("development_eval_configs", [])

        # Create a lookup for development_eval_configs by id
        eval_config_lookup = {
            config["id"]: {
                "display_name": config["display_name"],
                "eval_type_id": config["eval_type_id"],
            }
            for config in development_eval_configs
        }

        # Clean dataset rows
        for row in dataset_rows:
            if "dataset_eval_results" in row:
                for eval_result in row["dataset_eval_results"]:
                    eval_config_id = eval_result.get("development_eval_config_id")
                    if eval_config_id and eval_config_id in eval_config_lookup:
                        eval_result["development_eval_config"] = eval_config_lookup[
                            eval_config_id
                        ]
                    eval_result.pop("eval_run", None)
                    eval_result.pop("development_eval_config_id", None)

        cleaned_response = {
            "dataset": {
                "id": dataset.get("id"),
                "source": dataset.get("source"),
                "user_id": dataset.get("user_id"),
                "org_id": dataset.get("org_id"),
                "workspace_slug": dataset.get("workspace_slug"),
                "name": dataset.get("name"),
                "description": dataset.get("description"),
                "language_model_id": dataset.get("language_model_id"),
                "prompt_template": dataset.get("prompt_template"),
                "reference_dataset_id": dataset.get("reference_dataset_id"),
                "created_at": dataset.get("created_at"),
                "updated_at": dataset.get("updated_at"),
                "reference_dataset": dataset.get("reference_dataset"),
                "derived_datasets": dataset.get("derived_datasets"),
                "datacolumn": dataset.get("datacolumn"),
            },
            "dataset_rows": dataset_rows,
        }

        return cleaned_response
