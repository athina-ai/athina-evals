from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Callable, Iterable
from athina.steps.base import Step


class Map(Step):
    """
    Step that applies a function to each item in the input data.

    Attributes:
        fn (Callable[[Any], Any]): Function to apply to each item.
    """

    fn: Callable[[Any], Any]

    def execute(self, input_data: Any) -> List[Any]:
        """Apply a function to each item in the input data."""
        if not isinstance(input_data, Iterable):
            raise ValueError("Input data must be an iterable")
        results = list(map(self.fn, input_data))
        return results
