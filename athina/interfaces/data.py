from typing import TypedDict


class DataPoint(TypedDict):
    """Data point for a single inference."""

    response: str