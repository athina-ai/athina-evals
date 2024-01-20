from typing import TypedDict, Optional


class DataPoint(TypedDict):
    """Data point for a single inference."""

    response: str