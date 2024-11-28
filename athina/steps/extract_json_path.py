from typing import Union, Dict, Any
from athina.steps import Step
import json
from jsonpath_ng import parse

class ExtractJsonPath(Step):
    """
    Step that extracts json path from text using the JsonPath provided to the step.

    Attributes:
        input_column: The row's column to extract JsonPath from.
        json_path: The JsonPath to extract from the text.
    """

    input_column: str
    json_path: str

    def execute(self, input_data: Any) -> Union[Dict[str, Any], None]:
        """Extract the JsonPath from the input data."""

        if input_data is None:
            input_data = {}

        if not isinstance(input_data, dict):
            raise TypeError("Input data must be a dictionary.")

        input_text = input_data.get(self.input_column, None)
        
        if input_text is None:
            return None

        try:
            if isinstance(input_text, dict) or isinstance(input_text, list):
                input_json = input_text
            elif isinstance(input_text, str):
                input_json =  json.loads(input_text)
            else:
                return None
            result = parse(self.json_path).find(input_json)
            
            if not result or len(result) == 0:
                result = None
            elif len(result) == 1:
                result = result[0].value
            else:
                result = [match.value for match in result]

            return {
                "status": "success",
                "data": result,
            }
        except Exception as e:
            return {
                "status": "error",
                "data": str(e),
            }
