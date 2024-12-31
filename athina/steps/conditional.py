from typing import Any, Dict, List, Optional
from athina.steps.base import Step
from jinja2 import Environment, UndefinedError
from pydantic import ConfigDict
import ast
import operator
from typing import Union, Callable


class SafeExpressionEvaluator:
    """A safe expression evaluator that only allows specific operations and literals."""

    OPERATORS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.And: operator.and_,
        ast.Or: operator.or_,
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.USub: operator.neg,
        ast.Not: operator.not_,
        ast.In: lambda x, y: x in y,
        ast.NotIn: lambda x, y: x not in y,
    }

    @classmethod
    def evaluate(cls, expr: str) -> Any:
        """Safely evaluate a string expression.

        Args:
            expr: String expression to evaluate

        Returns:
            Result of the evaluation

        Raises:
            ValueError: If the expression contains unsupported operations
        """
        try:
            tree = ast.parse(expr, mode="eval")
            if tree.body is None:
                raise ValueError("Empty expression")
            return cls._eval_node(tree.body)
        except Exception as e:
            raise ValueError(f"Invalid expression: {str(e)}")

    @classmethod
    def _eval_node(cls, node: ast.AST) -> Any:
        """Recursively evaluate an AST node."""

        # Handle literals
        if isinstance(node, ast.Constant):
            return node.value

        # Handle lists
        elif isinstance(node, ast.List):
            return [cls._eval_node(elt) for elt in node.elts]

        # Handle tuples
        elif isinstance(node, ast.Tuple):
            return tuple(cls._eval_node(elt) for elt in node.elts)

        # Handle dict literals
        elif isinstance(node, ast.Dict):
            return {
                cls._eval_node(k): cls._eval_node(v)
                for k, v in zip(node.keys, node.values)
            }

        # Handle comparison operations
        elif isinstance(node, ast.Compare):
            left = cls._eval_node(node.left)
            for op, comp in zip(node.ops, node.comparators):
                if not isinstance(op, tuple(cls.OPERATORS.keys())):
                    raise ValueError(f"Unsupported operator: {op.__class__.__name__}")
                right = cls._eval_node(comp)
                left = cls.OPERATORS[op.__class__](left, right)
            return left

        # Handle boolean operations
        elif isinstance(node, ast.BoolOp):
            values = [cls._eval_node(val) for val in node.values]
            if isinstance(node.op, ast.And):
                return all(values)
            elif isinstance(node.op, ast.Or):
                return any(values)
            raise ValueError(
                f"Unsupported boolean operator: {node.op.__class__.__name__}"
            )

        # Handle unary operations
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, tuple(cls.OPERATORS.keys())):
                raise ValueError(
                    f"Unsupported unary operator: {node.op.__class__.__name__}"
                )
            operand = cls._eval_node(node.operand)
            return cls.OPERATORS[node.op.__class__](operand)

        # Handle binary operations
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, tuple(cls.OPERATORS.keys())):
                raise ValueError(
                    f"Unsupported binary operator: {node.op.__class__.__name__}"
                )
            left = cls._eval_node(node.left)
            right = cls._eval_node(node.right)
            return cls.OPERATORS[node.op.__class__](left, right)

        raise ValueError(f"Unsupported expression type: {node.__class__.__name__}")


class ConditionalStep(Step):
    """Step that evaluates conditions and executes appropriate branch steps."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    branches: List[Dict]
    env: Environment = Environment(variable_start_string="{{", variable_end_string="}}")

    def _evaluate_condition(self, condition: str, context: Dict) -> bool:
        """Evaluate a Jinja condition with given context."""
        try:
            template = self.env.from_string(condition)
            rendered_condition = template.render(**context)
            return SafeExpressionEvaluator.evaluate(rendered_condition)
        except (UndefinedError, ValueError) as e:
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
