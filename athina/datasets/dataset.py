import json
from typing import Any, List, Optional
import requests
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
    name: str
    description: Optional[str] = None
    language_model_id: Optional[str] = None
    prompt_template: Optional[Any] = None
    rows: List[DatasetRow] = field(default_factory=list)

    @staticmethod
    def create(name: str, description: Optional[str] = None, language_model_id: Optional[str] = None, prompt_template: Optional[Any] = None, rows: List[DatasetRow] = None):
        """
        Creates a new dataset with the specified properties.
        Parameters:
        - name (str): The name of the dataset. This is a required field.
        - description (Optional[str]): An optional textual description of the dataset, providing additional context or metadata.
        - language_model_id (Optional[str]): An optional identifier for the language model associated with this dataset.
        - prompt_template (Optional[Any]): An optional template for prompts used in this dataset.

        Returns:
        The newly created dataset object

        Raises:
        - Exception: If the dataset could not be created due to an error like invalid parameters, database errors, etc.
        """
        dataset = Dataset(name=name, description=description, language_model_id=language_model_id, prompt_template=prompt_template, rows=rows)
        created_dataset = AthinaApiService.create_dataset(dataset)
        return created_dataset
       