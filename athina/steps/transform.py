from typing import Union, Dict, Iterable, Any
from athina.helpers.json import JsonExtractor
from athina.steps import Step


class ExtractJsonFromString(Step):
    """
    Step that extracts JSON data from a string.
    """

    def execute(
        self, input_data: str
    ) -> Union[Dict[str, Any], Iterable[Dict[str, Any]]]:
        """Extract JSON data from the input string."""

        if (
            input_data is None
            or not isinstance(input_data, str)
            or len(input_data) == 0
        ):
            raise TypeError("Input data must be a valid string.")

        output = JsonExtractor.extract_first_json_entity(input_data)

        if output is None:
            raise ValueError("No valid JSON data found in the input string.")
        return output
