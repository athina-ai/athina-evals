from typing import Any, Dict, List, Optional
from athina.steps.base import Step
from jinja2 import Environment, UndefinedError
from pydantic import ConfigDict


class ConditionalStep(Step):
    """
    Step that evaluates conditions and executes appropriate branch steps.

    Attributes:
        branches (List[Dict]): List of branch configurations with conditions and steps
        env (Environment): Jinja environment for condition evaluation
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    branches: List[Dict]
    env: Environment = Environment(variable_start_string="{{", variable_end_string="}}")

    def _evaluate_condition(self, condition: str, context: Dict) -> bool:
        """Evaluate a Jinja condition with given context."""
        try:
            template = self.env.from_string(condition)
            rendered_condition = template.render(**context)
            return eval(rendered_condition)  # Consider using a safer eval alternative
        except (UndefinedError, Exception) as e:
            print(f"Error evaluating condition: {str(e)}")
            return False

    def _execute_branch_steps(self, steps: List[Step], inputs: Dict) -> Dict:
        """Execute a sequence of steps with given inputs."""
        cumulative_context = inputs.copy()
        final_output = None
        executed_steps = []

        for step in steps:
            step_result = step.execute(cumulative_context)
            executed_steps.append(step_result)
            cumulative_context = {
                **cumulative_context,
                f"{step.name}": step_result.get("data", {}),
            }
            final_output = step_result.get("data")

        return {
            "status": "success",
            "data": final_output,
            "metadata": {"executed_steps": executed_steps},
        }

    def execute(self, inputs: Dict) -> Dict:
        """Execute the conditional step by evaluating branches and running appropriate steps."""
        try:
            # Find the first matching branch
            for branch in self.branches:
                branch_type = branch.get("branch_type")
                condition = branch.get("condition")

                if branch_type == "else" or (
                    condition and self._evaluate_condition(condition, inputs)
                ):
                    result = self._execute_branch_steps(branch.get("steps", []), inputs)
                    if result.get("status") == "success":
                        result["metadata"]["executed_branch"] = branch_type
                    return result

            return {
                "status": "error",
                "data": "No matching branch found",
                "metadata": {},
            }

        except Exception as e:
            return {
                "status": "error",
                "data": f"Conditional step execution failed: {str(e)}",
                "metadata": {},
            }
