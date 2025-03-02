from typing import Dict, List
from athina.steps.base import Step
from pydantic import ConfigDict
from athina.steps.code_execution_v2 import CodeExecutionV2, EXECUTION_E2B


class ConditionalStep(Step):
    """Step that evaluates conditions and executes appropriate branch steps."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    branches: List[Dict]

    def _evaluate_condition(self, condition: str, context: Dict) -> bool:
        """Evaluate a Python condition with given context using sandbox execution."""
        try:
            # Create evaluation code that returns a boolean
            evaluation_code = f"result = bool({condition})\nprint(result)"
            executor = CodeExecutionV2(
                code=evaluation_code,
                session_id=context.get("session_id", "default"),
                execution_environment=EXECUTION_E2B,
                sandbox_timeout=40,  # 15 sec timeout
            )

            result = executor.execute(context)

            if result["status"] == "error":
                print(f"Error evaluating condition: {result['data']}")
                return False
            return result["data"].strip().lower() == "true"

        except Exception as e:
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

    def execute(self, input_data: Dict) -> Dict:
        """Execute the conditional step by evaluating branches and running appropriate steps."""
        try:
            # Find the first matching branch
            for branch in self.branches:
                branch_type = branch.get("branch_type")
                condition = branch.get("condition")

                if branch_type == "else" or (
                    condition and self._evaluate_condition(condition, input_data)
                ):
                    result = self._execute_branch_steps(branch.get("steps", []), input_data)
                    if result.get("status") == "success":
                        result["metadata"]["executed_branch"] = {
                            "condition": condition,
                            "branch_type": branch_type,
                        }
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
