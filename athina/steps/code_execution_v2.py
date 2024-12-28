from typing import Union, Dict, Any, Optional, Literal, ClassVar, TypedDict
from athina.steps import Step
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from dotenv import load_dotenv
import time
import json

try:
    from e2b_code_interpreter import Sandbox

    HAS_E2B = True
except ImportError:
    HAS_E2B = False

# Load environment variables
load_dotenv()

# Constants
EXECUTION_LOCAL = "local"
EXECUTION_E2B = "e2b"
ExecutionEnvironment = Literal["local", "e2b"]

VARS_START_MARKER = "__VARS_START__"
VARS_END_MARKER = "__VARS_END__"
COMMAND_PREFIX = "!"


class StepResult(TypedDict):
    status: Literal["success", "error"]
    data: str
    metadata: Dict[str, Any]


# Extract variable serialization logic
def _serialize_variable(name: str, value: Any) -> Optional[str]:
    """
    Attempt to serialize a variable to a string representation.
    Returns None if serialization fails.
    """
    try:
        # For multi-line strings, use triple quotes and preserve indentation
        serialized_value = repr(value)
        if "\n" in serialized_value:
            # Remove any existing quotes and wrap in triple quotes
            clean_value = serialized_value.strip("'\"")
            serialized_value = f'"""{clean_value}"""'
        # Ensure the assignment is at root level (no indentation)
        return f"{name} = {serialized_value}"
    except Exception as e:
        print(f"Error serializing variable {name}: {str(e)}")
        return None


# Extract variable capture code into a constant
VARIABLE_CAPTURE_CODE = f"""
import json

_exported_vars = {{}}
_locals = locals()
_globals = globals()
_builtin_names = dir(__builtins__)

# Create a list of items to iterate over to prevent dictionary modification during iteration
_global_items = list(_globals.items())

for var_name, var_value in _global_items:
    if (not var_name.startswith('_') and
        var_name not in _builtin_names and
        var_name not in ['json']):
        try:
            json.dumps(var_value)  # Test if value is JSON serializable
            _exported_vars[var_name] = var_value
        except:
            print(f"Could not serialize {{var_name}}")
            continue

print('{VARS_START_MARKER}')
print(json.dumps(_exported_vars))
print('{VARS_END_MARKER}')
"""


class CodeExecutionV2(Step):
    """
    Step that executes code using either local environment or E2B sandbox.

    Attributes:
        code (str): The code to execute.
        session_id (str): Unique identifier for the sandbox session.
        name (Optional[str]): Name identifier for the execution.
        execution_environment (ExecutionEnvironment): Execution context ('local' or 'e2b').
        _sandbox (Optional[Sandbox]): E2B sandbox instance.
        DEFAULT_TIMEOUT (ClassVar[int]): Default timeout for sandbox operations.
        sandbox_timeout (Optional[int]): Custom timeout for sandbox operations.
    """

    code: str
    session_id: str
    name: Optional[str] = None
    _sandbox: Optional[Sandbox] = None
    execution_environment: ExecutionEnvironment = EXECUTION_LOCAL
    DEFAULT_TIMEOUT: ClassVar[int] = 60  # 1 minute default timeout for sandbox
    MAX_TIMEOUT: ClassVar[int] = 300  # 5 minute limit for e2b sandbox execution
    sandbox_timeout: Optional[int] = None

    def __init__(
        self,
        execution_environment: ExecutionEnvironment = EXECUTION_LOCAL,
        sandbox_timeout: Optional[int] = None,
        **data,
    ):
        super().__init__(**data)
        self._sandbox = None
        self.execution_environment = execution_environment
        self.sandbox_timeout = sandbox_timeout

    def _create_or_initialize_sandbox(self):
        """Checks if sandbox exists and connects to it or creates a new one if not"""
        if not self.session_id:
            raise ValueError("session_id is required for e2b execution")

        try:
            running_sandboxes = Sandbox.list()

            for sandbox in running_sandboxes:
                if sandbox.metadata.get("session_id") == self.session_id:
                    # Connect to the existing sandbox
                    self._sandbox = Sandbox.connect(sandbox.sandbox_id)
                    break

            if self._sandbox is None:
                self._sandbox = Sandbox(
                    timeout=min(
                        self.sandbox_timeout or self.DEFAULT_TIMEOUT, self.MAX_TIMEOUT
                    ),
                    metadata={"session_id": self.session_id},
                )
                print(f"Created new sandbox with ID: {self._sandbox.sandbox_id}")

        except Exception as e:
            print(f"Error initializing sandbox: {str(e)}")
            raise RuntimeError(f"Failed to initialize sandbox: {str(e)}") from e

    def _create_step_result(
        self,
        status: Literal["success", "error"],
        data: str,
        start_time: float,
        exported_vars: Optional[Dict] = None,
    ) -> StepResult:
        """
        Create a standardized result object for step execution.

        Args:
            status: Execution status ("success" or "error")
            data: Output data or error message
            start_time: Time when execution started
            exported_vars: Optional dictionary of exported variables
        """
        execution_time_ms = round((time.time() - start_time) * 1000)
        metadata: Dict[str, Any] = {"response_time": execution_time_ms}

        if exported_vars is not None:
            metadata["exported_vars"] = exported_vars

        return {"status": status, "data": data, "metadata": metadata}

    def _execute_local(self, input_data: dict, start_time: float) -> StepResult:
        """Execute code locally using exec"""
        globals_dict = {"__builtins__": __builtins__}
        globals_dict.update(input_data)

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(self.code, globals_dict)

            return self._create_step_result(
                status="success", data=stdout_buffer.getvalue(), start_time=start_time
            )
        except Exception as e:
            return self._create_step_result(
                status="error",
                data=f"Failed to execute the code.\nDetails:\n{str(e)}",
                start_time=start_time,
            )

    def _prepare_input_variables(self, input_data: dict) -> list[str]:
        """
        Prepare input variables for sandbox execution.
        Returns a list of variable initialization statements.
        """
        input_vars_code = []

        for var_name, var_value in input_data.items():
            if isinstance(var_value, dict) and "exported_vars" in var_value:
                # Handle exported vars from previous steps
                for exp_var_name, exp_var_value in var_value["exported_vars"].items():
                    if code := _serialize_variable(exp_var_name, exp_var_value):
                        input_vars_code.append(code)
            else:
                if code := _serialize_variable(var_name, var_value):
                    input_vars_code.append(code)

        return input_vars_code

    def _extract_exported_vars(self, stdout: str) -> dict:
        """
        Extract exported variables from sandbox output.
        Returns empty dict if extraction fails.
        """
        try:
            vars_start = stdout.find(f"{VARS_START_MARKER}\n") + len(
                f"{VARS_START_MARKER}\n"
            )
            vars_end = stdout.find(f"\n{VARS_END_MARKER}")

            if vars_start > -1 and vars_end > -1:
                return json.loads(stdout[vars_start:vars_end])
        except Exception as e:
            print(f"Error extracting variables: {str(e)}")

        return {}

    def _execute_e2b(self, input_data: dict, start_time: float) -> StepResult:
        """
        Execute code in E2B sandbox.

        The execution follows these steps:
        1. Initialize/connect to sandbox
        2. Initialize input variables in sandbox
        3. Execute code (either as commands or Python)
        4. Capture and extract output variables for Python code
        """
        try:
            self._create_or_initialize_sandbox()
            if self._sandbox is None:
                print("Sandbox is not initialized")
                return self._create_step_result(
                    status="error",
                    data="Sandbox is not initialized",
                    start_time=start_time,
                )

            # Initialize input variables if we're running Python code
            if not self.code.strip().startswith(COMMAND_PREFIX):
                input_vars_code = self._prepare_input_variables(input_data)
                if input_vars_code:
                    setup_code = "\n".join(input_vars_code)
                    setup_execution = self._sandbox.run_code(setup_code)
                    if setup_execution.error:
                        print(
                            f"Error setting up input variables: {setup_execution.error}"
                        )

            # Execute code based on type (commands or Python)
            if self.code.strip().startswith(COMMAND_PREFIX):
                # Handle command execution
                commands = [
                    line.strip()[1:] for line in self.code.split("\n") if line.strip()
                ]
                output = []
                for command in commands:
                    command_result = self._sandbox.commands.run(command)
                    if command_result.error or command_result.exit_code != 0:
                        return self._create_step_result(
                            status="error",
                            data=f"Failed to execute command: {command}\nexit_code: {command_result.exit_code}\nDetails:\n{command_result.error}",
                            start_time=start_time,
                        )
                    print(f"Command output: {command_result}")
                    if command_result.stdout:
                        output.extend(command_result.stdout)
                return self._create_step_result(
                    status="success",
                    data="".join(output),
                    start_time=start_time,
                    exported_vars={},
                )
            else:
                # Handle Python code execution
                execution = self._sandbox.run_code(self.code)
                if execution.error:
                    return self._create_step_result(
                        status="error",
                        data=f"Failed to execute the code.\nDetails:\n{execution.error}",
                        start_time=start_time,
                    )

                # Capture variables for Python execution
                var_execution = self._sandbox.run_code(VARIABLE_CAPTURE_CODE)
                if var_execution.error:
                    print(f"Error capturing variables: {var_execution.error}")
                    return self._create_step_result(
                        status="success",
                        data="\n".join(execution.logs.stdout),
                        start_time=start_time,
                        exported_vars={},
                    )

                # Extract and return results
                exported_vars = self._extract_exported_vars(
                    "\n".join(var_execution.logs.stdout)
                )
                return self._create_step_result(
                    status="success",
                    data="\n".join(execution.logs.stdout),
                    start_time=start_time,
                    exported_vars=exported_vars,
                )

        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            return self._create_step_result(
                status="error",
                data=f"Failed to execute the code.\nDetails:\n{str(e)}",
                start_time=start_time,
            )

    def execute(self, input_data: Any) -> StepResult:
        """
        Execute the code with the input data.

        Args:
            input_data: Dictionary containing input variables for code execution.

        Returns:
            Dict containing execution status, output data, and metadata.

        Raises:
            TypeError: If input_data is not a dictionary.
            ValueError: If session_id is empty in e2b mode.
        """

        if not self.code.strip():
            raise ValueError("No code provided for execution")

        if self.execution_environment == "e2b" and not self.session_id:
            raise ValueError("session_id is required for e2b execution")

        input_data = input_data or {}
        if not isinstance(input_data, dict):
            raise TypeError("Input data must be a dictionary")

        # Start timing
        start_time = time.time()

        if self.execution_environment == "e2b":
            if not HAS_E2B:
                print("Warning: e2b not installed, falling back to local execution")
                return self._execute_local(input_data, start_time)
            return self._execute_e2b(input_data=input_data, start_time=start_time)
        else:
            return self._execute_local(input_data, start_time)
