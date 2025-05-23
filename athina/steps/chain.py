from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
from athina.steps.base import Step
from athina.llms.abstract_llm_service import AbstractLlmService
import json


class Chain(BaseModel):
    """
    A sequence of steps to be executed in order.

    Attributes:
        sequence (List[Step]): The sequence of steps to execute.
        context (Dict[str, Any]): The context shared across steps.
    """

    sequence: List[Step]
    context: Dict[str, Any] = {}

    def run(self, inputs: Dict[str, Any]) -> "Chain":
        """Run the sequence of steps with the provided inputs."""
        self.context = inputs
        history = []
        for step in self.sequence:
            if self.context.get("__return__", False):
                break
            history = self.context.get("__steps__", [])
            current_step_output = step.run(context=self.context, history=history)
            if step.output_key is not None:
                self.context[step.output_key] = current_step_output
            self.context["__steps__"] = history
        return self

    def get_context(self) -> Dict[str, Any]:
        """Get the current context."""
        return self.context

    def get_output(self, key: Optional[str] = None) -> Any:
        """Get the output of the last step or a specific output key."""
        if key is None:
            last_step = (
                self.context.get("__steps__", [])[-1]
                if self.context.get("__steps__", [])
                else None
            )
            return (
                last_step.get("output", None)
                if last_step and isinstance(last_step, dict)
                else None
            )
        return self.context.get(key, None)

    def execute(self, input_data: Any) -> Union[Dict[str, Any], None]:
        """Execute the sequence of steps with the provided inputs."""
        cumulative_context = input_data.copy()
        emptyStep = Step()
        prepared_body = emptyStep.prepare_dict(self.context, input_data)
        cumulative_context = {**cumulative_context, **prepared_body}
        latest_step_output = None
        all_steps_output= {}
        for step in self.sequence:
            step_output = step.execute(input_data=cumulative_context)
            exported_vars = step_output.get("metadata", {}).get("exported_vars", {})
            if step.name:
                cumulative_context={
                    **cumulative_context,
                    **exported_vars,
                    f'{step.name}_str': isinstance(step_output.get("data"), dict) and json.dumps(step_output.get("data")) or None,
                    step.name: step_output.get("data")
                }
                all_steps_output = {
                    **all_steps_output,
                    step.name: step_output
                }
            latest_step_output = step_output
        response = {
            "chain_output": latest_step_output,
            "all_steps_output": all_steps_output,
        }
        return response
