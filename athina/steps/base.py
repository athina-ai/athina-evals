import os
import json
import logging
from typing import Dict, Any, List, Iterable, Optional, Callable
from pydantic import BaseModel
from athina.helpers.json import JsonHelper, JsonExtractor
from athina.llms.abstract_llm_service import AbstractLlmService
from athina.llms.openai_service import OpenAiService
from athina.keys import OpenAiApiKey
import functools


# Configure logging
log_level = os.getenv("LOG_LEVEL", logging.INFO)
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)


class StepError(Exception):
    """Custom exception for errors in steps."""

    pass


def step(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        context = kwargs.get("context", {})
        history = kwargs.get("history", [])
        try:
            input_data = self.extract_input_data(context)
            logger.debug(
                f"Running {self.__class__.__name__} with input data: {input_data}"
            )
            result = func(self, input_data=input_data, context=context, history=history)
            logger.debug(f"Completed {self.__class__.__name__} with result: {result}")
            if self.output_key:
                context[self.output_key] = result
            return result
        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}: {e}", exc_info=True)
            history.append({"step": self.__class__.__name__, "error": str(e)})
            raise StepError(f"Error in {self.__class__.__name__}: {e}")

    return wrapper


class Step(BaseModel):
    """
    Base class for all steps in a chain.

    Attributes:
        input_key (Optional[str]): Key to fetch the input data from the context.
        output_key (Optional[str]): Key to store the output data in the context.
        input_data (Optional[Any]): Direct input data for the step.
    """

    input_key: Optional[str] = None
    output_key: Optional[str] = None
    input_data: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary."""
        return self.model_dump()

    def extract_input_data(self, context: Dict[str, Any]) -> Any:
        """
        Extract the input data from the context or use the direct input data.

        Args:
            context (Dict[str, Any]): The context dictionary containing input data.

        Returns:
            Any: The extracted input data.
        """
        input_data = context.get(self.input_key, self.input_data)
        if (input_data is None or not isinstance(input_data, dict)) and self.input_key:
            input_data = context.get(self.input_key, self.input_data)
        else:
            input_data = context
        return input_data

    @step
    def run(
        self,
        context: Dict[str, Any],
        history: List[Dict[str, Any]],
        input_data: Optional[Any],
    ) -> Any:
        """Run the step with the provided context and history."""
        result = self.execute(input_data)
        if self.output_key:
            context[self.output_key] = result
        history.append({"step": self.__class__.__name__, "output": result})
        return result

    def execute(self, input_data: Any) -> Any:
        """Execute the core logic of the step. This should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement this method")


class Debug(Step):
    """
    Step that logs the context for debugging.

    Attributes:
        message (Optional[str]): Optional debug message to log.
    """

    message: Optional[str] = None

    def run(self, context: Dict[str, Any], history: List[Dict[str, Any]]) -> Any:
        """Run the step with the provided context and history."""
        logger.debug("DEBUG: ", json.dumps(context, indent=2))
        self.execute(context)
        history.append({"step": self.__class__.__name__, "output": None})
        return None

    def execute(self, input_data: Any) -> None:
        """Log the context for debugging."""
        if self.message:
            logger.debug(f"DEBUG: {self.message}")


class Fn(Step):
    """
    Step that runs a custom function with the input data.

    Attributes:
        fn (Callable[[Any, Dict[str, Any]], Any]): Custom function to run.
    """

    fn: Callable

    def execute(self, input_data: Any) -> Any:
        """Run a custom function with the input data."""
        result = self.fn(input_data)
        return result
