from typing import List, Optional
from athina.interfaces.athina import AthinaFilters
from athina.services.athina_api_service import AthinaApiService
from .loader import Loader, DataPoint


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
    ):
        """
        Initializes the loader with specified or default column names.
        """
        self.col_response = col_response
        self._raw_dataset = {}
        self._processed_dataset: List[DataPoint] = []

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
