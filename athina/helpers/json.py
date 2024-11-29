import json
import jsonschema
from jsonpath_ng import parse
from jsonschema import validate
from typing import Any, Optional


class JsonHelper:
    @staticmethod
    def _extract_json(data_string: str) -> str:
        """
        Extracts a JSON string from a larger string.
        Assumes the JSON content starts with '{' and continues to the end of the input string.
        """
        try:
            start_index = data_string.index("{")
            end_index = data_string.rfind("}")
            json_string = data_string[start_index : end_index + 1]
        except Exception as e:
            json_string = data_string
        return json_string

    @staticmethod
    def _load_json_from_text(text):
        """
        Extracts and loads a JSON string from a given text.
        """
        try:
            data = json.loads(text)
        except json.decoder.JSONDecodeError:
            raise ValueError("Failed to load JSON from text")
        return data

    @staticmethod
    def extract_json_from_text(text):
        # In case you cannot handle an error, return None
        if text is None:
            return None
        response_json_format = JsonHelper._extract_json(text)
        response_json = JsonHelper._load_json_from_text(response_json_format)
        return response_json


def validate_json(json_data, schema):
    try:
        validate(instance=json_data, schema=schema)
        return True, None
    except jsonschema.exceptions.ValidationError as err:
        return False, str(err)


def extract_json_path(json_data, json_path):
    try:
        jsonpath_expr = parse(json_path)
        match = jsonpath_expr.find(json_data)
        return [match.value for match in match] if match else None
    except Exception as e:
        return None


# New and improved JsonExtractor
# - can extract top-level arrays as well
# - uses stack based approach
class JsonExtractor:
    @staticmethod
    def extract_first_json_entity(text: str) -> Optional[Any]:
        """
        Extracts the first top-level JSON entity from a given text string.

        Args:
            text (str): The input text containing JSON entities.

        Returns:
            dict or list: The first JSON object or array extracted from the text, or None if no valid JSON is found.
        """
        i = 0
        length = len(text)

        while i < length:
            if text[i] in "{[":
                start_idx = i
                stack = [text[i]]
                i += 1

                while i < length and stack:
                    if text[i] in "{[":
                        stack.append(text[i])
                    elif text[i] in "}]":
                        stack.pop()
                    i += 1

                if not stack:
                    json_str = text[start_idx:i]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        continue
            else:
                i += 1

        return None
