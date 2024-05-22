from typing import Callable, Any, Dict, Optional

from athina.steps.base import Step


class Assert(Step):
    """
    Step that asserts a condition and raises an error if it fails.

    Attributes:
        condition (Callable[[Any, Dict[str, Any]], bool]): Function to evaluate the assertion.
        error_message (str): Error message to raise if the assertion fails.
    """

    condition: Callable[[Any], bool]
    error_message: str

    def execute(self, input_data: Any) -> Any:
        """Assert a condition and raise error if it fails."""
        if not self.condition(input_data):
            raise AssertionError(f"Assertion failed - {self.error_message}")
        return input_data


class If(Step):
    """
    Step that conditionally runs either the 'then' or 'else' step based on a condition.

    Attributes:
        condition (Callable[[Any], bool]): Function to evaluate the condition.
        then_step (Step): Step to run if the condition is true.
        else_step (Optional[Step]): Step to run if the condition is false.
    """

    condition: Callable[[Any], bool]
    then_step: Step
    else_step: Optional[Step] = None

    def execute(self, input_data: Any) -> Any:
        """Run the 'then' step if the condition is true, else run the 'else' step."""
        result = (
            self.then_step.execute(input_data)
            if self.condition(input_data)
            else self.else_step.execute(input_data) if self.else_step else None
        )
        return result
