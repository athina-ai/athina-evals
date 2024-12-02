from typing import Union, Dict, Any, Optional, Literal, ClassVar
from athina.steps import Step
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from dotenv import load_dotenv
from e2b_code_interpreter import Sandbox
import time
import json

# Load environment variables
load_dotenv()

ExecutionEnvironment = Literal["local", "e2b"]


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
    execution_environment: ExecutionEnvironment = "e2b"
    DEFAULT_TIMEOUT: ClassVar[int] = 60  # 1 minute default timeout for sandbox
    MAX_TIMEOUT: ClassVar[int] = 300  # 5 minute limit for e2b sandbox execution
    sandbox_timeout: Optional[int] = None

    def __init__(
        self,
        execution_environment: ExecutionEnvironment = "e2b",
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
                if self.code.startswith("!"):
                    # Run the code as a command
                    commands = map(lambda x: x[1:], self.code.split("\n"))
                    for command in commands:
                        self._sandbox.commands.run(command)
                print(f"Created new sandbox with ID: {self._sandbox.sandbox_id}")

        except Exception as e:
            print(f"Error initializing sandbox: {str(e)}")
            raise RuntimeError(f"Failed to initialize sandbox: {str(e)}") from e

    def _execute_local(self, input_data: dict, start_time: float) -> Dict[str, Any]:
        """Execute code locally using exec"""
        # Create a new dictionary for globals, starting with a clean __builtins__
        globals_dict = {"__builtins__": __builtins__}
        globals_dict.update(input_data)

        # Create string buffers to capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        try:
            # Capture both stdout and stderr during execution
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(self.code, globals_dict)

            execution_time_ms = round((time.time() - start_time) * 1000)

            return {
                "status": "success",
                "data": stdout_buffer.getvalue(),
                "metadata": {"response_time": execution_time_ms},
            }
        except Exception as e:
            execution_time_ms = round((time.time() - start_time) * 1000)
            return {
                "status": "error",
                "data": f"Failed to execute the code.\nDetails:\n{str(e)}",
                "metadata": {"response_time": execution_time_ms},
            }

    def _execute_e2b(self, input_data: dict, start_time: float) -> Dict[str, Any]:
        """Execute code in E2B sandbox"""
        try:
            # Ensure we have the correct sandbox
            self._create_or_initialize_sandbox()

            if self._sandbox is None:
                return {
                    "status": "error",
                    "data": "Sandbox is not initialized",
                    "metadata": {"response_time": 0},
                }

            # Split code into commands and Python code
            lines = self.code.split("\n")
            commands = []
            python_code = []

            for line in lines:
                if line.strip().startswith("!"):
                    commands.append(line.strip()[1:])  # Remove the '!'
                else:
                    python_code.append(line)

            if python_code:
                # Prepare input variables initialization code
                input_vars_code = []
                for var_name, var_value in input_data.items():
                    if isinstance(var_value, dict) and "exported_vars" in var_value:
                        # Include exported vars directly
                        for exp_var_name, exp_var_value in var_value[
                            "exported_vars"
                        ].items():
                            try:
                                serialized_value = repr(exp_var_value)
                                input_vars_code.append(
                                    f"{exp_var_name} = {serialized_value}"
                                )
                            except Exception as e:
                                print(
                                    f"Error serializing exported var {exp_var_name}: {str(e)}"
                                )
                    else:
                        try:
                            serialized_value = repr(var_value)
                            input_vars_code.append(f"{var_name} = {serialized_value}")
                        except Exception as e:
                            print(f"Error serializing input var {var_name}: {str(e)}")

                # First, initialize input variables
                if input_vars_code:
                    setup_code = "\n".join(input_vars_code)
                    setup_execution = self._sandbox.run_code(setup_code)
                    if setup_execution.error:
                        print(
                            f"Error setting up input variables: {setup_execution.error}"
                        )
                    else:
                        print("Input variables initialized successfully")

                # Then, run the actual code
                main_code = "\n".join(python_code)
                execution = self._sandbox.run_code(main_code)

                if execution.error:
                    print(f"\nExecution error: {execution.error}")
                    execution_time_ms = round((time.time() - start_time) * 1000)
                    return {
                        "status": "error",
                        "data": f"Failed to execute the code.\nDetails:\n{execution.error}",
                        "metadata": {"response_time": execution_time_ms},
                    }

                # Finally, run the variable capture code
                capture_vars_code = """
import json

_exported_vars = {}
_locals = locals()
_globals = globals()
_builtin_names = dir(__builtins__)

for var_name, var_value in _globals.items():
    if (not var_name.startswith('_') and 
        var_name not in _builtin_names and 
        var_name not in ['json']):
        try:
            json.dumps(var_value)  # Test if value is JSON serializable
            _exported_vars[var_name] = var_value
        except:
            print(f"Could not serialize {var_name}")
            continue

print('__VARS__START__')
print(json.dumps(_exported_vars))
print('__VARS__END__')
"""

                # Execute variable capture
                var_execution = self._sandbox.run_code(capture_vars_code)
                execution_time_ms = round((time.time() - start_time) * 1000)

                if var_execution.error:
                    print(f"Error during variable capture: {var_execution.error}")
                    return {
                        "status": "success",
                        "data": "\n".join(execution.logs.stdout),
                        "metadata": {
                            "response_time": execution_time_ms,
                            "exported_vars": {},
                        },
                    }

                # Extract variables from output
                stdout = "\n".join(var_execution.logs.stdout)
                try:
                    vars_start = stdout.find("__VARS__START__\n") + len(
                        "__VARS__START__\n"
                    )
                    vars_end = stdout.find("\n__VARS__END__")
                    if vars_start > -1 and vars_end > -1:
                        exported_vars = json.loads(stdout[vars_start:vars_end])
                    else:
                        exported_vars = {}
                except Exception as e:
                    print(f"Error extracting variables: {str(e)}")
                    exported_vars = {}

                return {
                    "status": "success",
                    "data": "\n".join(execution.logs.stdout),
                    "metadata": {
                        "response_time": execution_time_ms,
                        "exported_vars": exported_vars,
                    },
                }

            # If only commands were executed, return success
            execution_time_ms = round((time.time() - start_time) * 1000)
            return {
                "status": "success",
                "data": "Commands executed successfully",
                "metadata": {
                    "response_time": execution_time_ms,
                    "exported_vars": {},
                },
            }

        except Exception as e:
            print(f"\nUnexpected error: {str(e)}")
            execution_time_ms = round((time.time() - start_time) * 1000)
            return {
                "status": "error",
                "data": f"Failed to execute the code.\nDetails:\n{str(e)}",
                "metadata": {"response_time": execution_time_ms},
            }

    def execute(self, input_data: Any) -> Dict[str, Any]:
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

        if self.execution_environment == "local":
            return self._execute_local(input_data, start_time)
        else:
            return self._execute_e2b(input_data=input_data, start_time=start_time)
