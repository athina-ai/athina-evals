import json
from typing import Dict, Any, Union

allowed_metadata_keys = [
    "content_type",
    "file_name",
    "file_size",
    "chart_type",
    "title",
    "x_axis_key",
    "data_keys",
    "height",
    "colors",
]


def get_filtered_metadata(data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Extract essential metadata from data, which can be either a dictionary or a JSON string.

    Args:
        data: Input data, either as dict or JSON string

    Returns:
        Updated metadata dictionary
    """
    # Handle case where data is a JSON string
    if isinstance(data, str):
        try:
            data = data.strip()
            data = json.loads(data)
        except json.JSONDecodeError:
            try:
                # Try to remove the JSON markers and load the remaining string
                data = data.replace("```json", "").replace("```", "").strip()
                data = json.loads(data)
            except json.JSONDecodeError:
                # Not a valid JSON string, return empty metadata
                return {}

    # Now handle dictionary data
    if isinstance(data, dict) and "metadata" in data:
        metadata = data["metadata"]
        filtered_metadata = {
            k: v for k, v in metadata.items() if k in allowed_metadata_keys
        }
        return filtered_metadata

    return {}
