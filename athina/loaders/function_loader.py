from typing import List, Optional

from athina.helpers.function_eval_util import get_named_non_default_parameters
from athina.interfaces.athina import AthinaFilters
from athina.interfaces.data import DataPoint
from .loader import Loader
from athina.evals.function.functions import operations


class FunctionEvalDataPoint(DataPoint):
    """Data point for a single Function invocation."""
    keywords: Optional[List[str]]
    response: str


class FunctionEvalLoader(Loader):
    """
    This class is a data loader for running function evals on datasets.

    Attributes:
        function_name (str): The name of the evaluator function to be used.
        raw_dataset (dict): The raw dataset as loaded from the source.
        processed_dataset (list): The processed dataset with queries, context, response and other attributes if present.
    """

    def __init__(
        self,
        function_name="contains_any",
    ):
        """
        Initializes the loader with specified or default column names.
        """
        self.function_name = function_name
        self.col_response = "response"
        self._raw_dataset = {}
        self._processed_dataset: List[FunctionEvalDataPoint] = []

    def process(self) -> None:
        """
        Transforms the raw data into a structured format. Processes each entry from the raw dataset, and extracts attributes.

        Raises:
            KeyError: If mandatory columns (query, context or response) are missing in the raw dataset.
            TypeError: If context is not a list of strings.
        """
        operator = operations.get(self.function_name)
        non_default_parameters = get_named_non_default_parameters(operator)
        for raw_instance in self._raw_dataset:
            # Check for mandatory columns in raw_instance 

            if self.col_response not in raw_instance:
                raise KeyError(f"'{self.col_response}' not found in provided data.")
            for parameter in non_default_parameters:
                if parameter not in raw_instance:
                    raise KeyError(f"'{parameter}' not found in provided data.")
            # Create a processed instance with mandatory fields
            processed_instance = {
                "response": raw_instance[self.col_response],
            }
            for parameter in non_default_parameters:
                processed_instance[parameter] = raw_instance[parameter]
            # Store the results
            self._processed_dataset.append(processed_instance)