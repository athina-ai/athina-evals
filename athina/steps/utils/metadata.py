from typing import Dict, Any


def filter_essential_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out non-essential metadata keys."""
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
    for allowed_metadata_key in allowed_metadata_keys:
        if allowed_metadata_key in metadata:
            metadata[allowed_metadata_key] = metadata[allowed_metadata_key]

    return metadata
