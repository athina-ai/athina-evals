import json


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
