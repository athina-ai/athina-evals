from typing import List, TypedDict, Optional
from athina.interfaces.athina import AthinaFilters
from athina.services.athina_api_service import AthinaApiService
from .loader import Loader, DataPoint
import json


class ResponseLoader(Loader):
    """
    This class is a data loader for evals that only evaluate the response.

    Attributes:
        col_response (str): The column name corresponding to the response.
        raw_dataset (dict): The raw dataset as loaded from the source.
        processed_dataset (list): The processed dataset with responses.
    """

    def __init__(
        self,
        col_response="response",
        format="json",
    ):
        """
        Initializes the loader with specified or default column names.
        """
        self.col_response = col_response
        self._raw_dataset = {}
        self._processed_dataset: List[DataPoint] = []
        self.format = format

    @property
    def processed_dataset(self) -> List[DataPoint]:
        """
        Returns the processed dataset.
        """
        return self._processed_dataset

    @property
    def raw_dataset(self):
        """
        Returns the raw dataset.
        """
        return self._raw_dataset

    def process(self) -> None:
        """
        Transforms the raw data into a structured format. Processes each entry from the raw dataset, and extracts attributes.

        Raises:
            KeyError: If mandatory columns (response) are missing in the raw dataset.
        """
        for raw_instance in self._raw_dataset:
            # Check for mandatory columns in raw_instance
            if self.col_response not in raw_instance:
                raise KeyError(f"'{self.col_response}' not found in provided data.")
            # Create a processed instance with mandatory fields
            processed_instance = {
                "response": raw_instance[self.col_response],
            }

            # Store the results
            self._processed_dataset.append(processed_instance)

    def load_json(self, filename: str) -> List[DataPoint]:
        """
        Loads and processes data from a JSON file.

        Raises:
            FileNotFoundError: If the specified JSON file is not found.
            json.JSONDecodeError: If there's an issue decoding the JSON.
        """
        try:
            with open(filename, "r") as f:
                self._raw_dataset = json.load(f)
                self.process()
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading JSON: {e}")

    def load_dict(self, data: list) -> List[DataPoint]:
        """
        Loads and processes data from a list of dictionaries.
        """
        self._raw_dataset = data
        self.process()
        return self._processed_dataset

    def load(self, data: list) -> List[DataPoint]:
        """
        Loads data based on the format specified.
        """
        if self.format == "json":
            return self.load_json(data)
        elif self.format == "dict":
            return self.load_dict(data)
        else:
            raise NotImplementedError("This file format has not been supported yet.")

    def load_athina_inferences(
        self,
        filters: Optional[AthinaFilters] = None,
        limit: Optional[int] = None,
    ) -> List[DataPoint]:
        """
        Load data from Athina API.
        By default, this will fetch the last 50 inferences from the API.
        """
        athina_inferences = AthinaApiService.load_inferences(filters, limit=limit)
        self._raw_dataset = list(
            map(
                lambda x: {
                    "response": x.prompt_response,
                },
                athina_inferences,
            )
        )
        self.process()
        return self._processed_dataset

    def load_csv(self) -> List[DataPoint]:
        """
        Placeholder for loading data from a CSV file.

        Raises:
            NotImplementedError: This method has not been implemented yet.
        """
        raise NotImplementedError("This method has not been implemented yet.")

    def load_pandas(self) -> List[DataPoint]:
        """
        Placeholder for loading data from a pandas DataFrame.

        Raises:
            NotImplementedError: This method has not been implemented yet.
        """
        raise NotImplementedError("This method has not been implemented yet.")
