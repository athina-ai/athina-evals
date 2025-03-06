import os
import json
import logging
from typing import Dict, Any, List, Iterable, Optional, Callable, TypedDict, Literal
from pydantic import BaseModel
from jinja2 import Environment
from athina.helpers.jinja_helper import PreserveUndefined
from athina.helpers.json import JsonHelper, JsonExtractor
from athina.llms.abstract_llm_service import AbstractLlmService
from athina.llms.openai_service import OpenAiService
from athina.keys import OpenAiApiKey
from athina.steps.utils.metadata import get_filtered_metadata
import functools
import time


# Configure logging
log_level = os.getenv("LOG_LEVEL", logging.INFO)
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)


class StepError(Exception):
    """Custom exception for errors in steps."""

    pass


class StepResult(TypedDict):
    status: Literal["success", "error"]
    data: str
    metadata: Dict[str, Any]


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
    name: Optional[str] = None

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

    def prepare_dict(
        self, object: Optional[Dict[str, Any]], input_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Prepare request body by rendering Jinja2 template."""
        if object is None:
            return None

        obj = json.dumps(object)
        env = self._create_jinja_env()
        prepared_obj = env.from_string(obj).render(**input_data)
        return json.loads(prepared_obj)

    def _create_step_result(
        self,
        status: Literal["success", "error"],
        data: Any,
        start_time: float,
        metadata: Dict[str, Any] = {},
        exported_vars: Optional[Dict] = None,
    ) -> StepResult:
        """
        Create a standardized result object for step execution.

        Args:
            status: Step execution status ("success" or "error")
            data: Output data or error message
            start_time: Time when step started execution (from perf_counter)
            metadata: Optional dictionary of metadata
            exported_vars: Optional dictionary of exported variables
        """
        metadata.update(get_filtered_metadata(data))

        if "response_time" not in metadata:
            execution_time_ms = round((time.perf_counter() - start_time) * 1000)
            metadata["response_time"] = execution_time_ms

        if exported_vars is not None:
            metadata["exported_vars"] = exported_vars

        return {"status": status, "data": data, "metadata": metadata}

    def _create_jinja_env(
        self,
        variable_start_string: str = "{{",
        variable_end_string: str = "}}",
    ) -> Environment:
        """Create a Jinja2 environment with custom settings."""
        return Environment(
            variable_start_string=variable_start_string,
            variable_end_string=variable_end_string,
            undefined=PreserveUndefined,
        )

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

    async def execute_async(self, input_data: Any) -> Any:
        """Execute the core logic of the step asynchronously. This should be implemented by subclasses."""
        pass


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
