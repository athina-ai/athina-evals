from typing import Any, List, Optional
from dataclasses import dataclass, field
from athina.services.athina_api_service import AthinaApiService


@dataclass
class DatasetRow:
    query: Optional[str] = None
    context: Optional[List[str]] = None
    response: Optional[str] = None
    expected_response: Optional[str] = None


@dataclass
class Dataset:
    id: str
    source: str
    name: str
    description: Optional[str] = None
    language_model_id: Optional[str] = None
    prompt_template: Optional[Any] = None
    rows: List[DatasetRow] = field(default_factory=list)

    @staticmethod
    def create(
        name: str,
        description: Optional[str] = None,
        language_model_id: Optional[str] = None,
        prompt_template: Optional[Any] = None,
        rows: List[DatasetRow] = None,
    ):
        """
        Creates a new dataset with the specified properties.
        Parameters:
        - name (str): The name of the dataset. This is a required field.
        - description (Optional[str]): An optional textual description of the dataset, providing additional context.
        - language_model_id (Optional[str]): An optional identifier for the language model associated with this dataset.
        - prompt_template (Optional[Any]): An optional template for prompts used in this dataset.

        Returns:
        The newly created dataset object

        Raises:
        - Exception: If the dataset could not be created due to an error like invalid parameters, database errors, etc.
        """
        dataset_data = {
            "source": "dev_sdk",
            "name": name,
            "description": description,
            "language_model_id": language_model_id,
            "prompt_template": prompt_template,
            "dataset_rows": rows or [],
        }

        # Remove keys where the value is None
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
    def add_rows(dataset_id: str, rows: List[DatasetRow]):
        """
        Adds rows to a dataset in batches of 100.

        Parameters:
        - dataset_id (str): The ID of the dataset to add rows to.
        - rows (List[DatasetRow]): The rows to add to the dataset.

        Raises:
        - Exception: If the API returns an error or the limit of 1000 rows is exceeded.
        """
        batch_size = 100
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            try:
                AthinaApiService.add_dataset_rows(dataset_id, batch)
            except Exception as e:
                raise

    @staticmethod
    def fetch_dataset_rows(dataset_id: str, number_of_rows: Optional[int] = None):
        """
        Fetches the rows of a dataset.

        Parameters:
        - dataset_id (str): The ID of the dataset to fetch rows.
        """
        return AthinaApiService.fetch_dataset_rows(dataset_id, number_of_rows)

    @staticmethod
    def dataset_link(dataset_id: str):
        return f"https://app.athina.ai/develop/{dataset_id}"
