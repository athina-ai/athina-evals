from typing import Union, Dict, Any
from athina.steps import Step
import subprocess
import os
import json
import tempfile 

class CodeExecution(Step):
    """
    Step that executes code using the code provided to the step.

    Attributes:
        code: The code to execute.
    """

    code: str

    def bandit_check(self, code: str) -> None:
        """
        Run Bandit security check on the provided code.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
            temp_file.write(code.encode('utf-8'))
            temp_file_path = temp_file.name
        try:
            result = subprocess.run(
                ["bandit", "-r", temp_file_path, "-f", "json", "-c", "bandit.yml"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                return json.dumps(result.stdout)
        finally:
            os.remove(temp_file_path)
        return None
    
    def execute(self, input_data: Any) -> Union[Dict[str, Any], None]:
        """Extract the JsonPath from the input data."""

        if input_data is None:
            input_data = {}

        if not isinstance(input_data, dict):
            raise TypeError("Input data must be a dictionary.")

        try:
            issues = self.bandit_check(self.code)
            if not issues:
                return { "error": "Security check failed. " + issues }
            from RestrictedPython import compile_restricted
            from RestrictedPython import safe_globals
            from RestrictedPython.Guards import safe_builtins
            from RestrictedPython.Eval import default_guarded_getitem, default_guarded_getiter

            custom_builtins = safe_builtins.copy()
            custom_builtins['type'] = type
            custom_builtins['__import__'] = __import__
            custom_globals = safe_globals.copy()
            custom_globals.update({
                '__builtins__': custom_builtins,
                'json': json,
                '_getitem_': default_guarded_getitem,
                '_getiter_': default_guarded_getiter
            })
            # Whitelist of allowed modules
            allowed_modules = {'json'}
            def guarded_import(name, *args, **kwargs):
                if name not in allowed_modules:
                    raise ImportError(f"Importing '{name}' is not allowed")
                return __import__(name, *args, **kwargs)

            custom_builtins['__import__'] = guarded_import
            loc = {}
            byte_code = compile_restricted(self.code, '<inline>', 'exec')
            exec(byte_code, custom_globals, loc)
            result = loc['main'](**input_data)
        except Exception as e:
            result = { "error": str(e) }

        return result
