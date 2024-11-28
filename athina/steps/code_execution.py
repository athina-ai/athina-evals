from typing import Union, Dict, Any, Optional
from athina.steps import Step
import subprocess
import os
import json
import re
import tempfile


class CodeExecution(Step):
    """
    Step that executes code using the code provided to the step.

    Attributes:
        code: The code to execute.
    """

    code: str
    name: Optional[str] = None

    def bandit_check(self, code: str) -> None:
        """
        Run Bandit security check on the provided code.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
            temp_file.write(code.encode("utf-8"))
            temp_file_path = temp_file.name
        try:
            result = subprocess.run(
                ["bandit", "-r", temp_file_path, "-f", "json", "-c", "bandit.yml"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return json.dumps(result.stdout)
        finally:
            os.remove(temp_file_path)
        return None

    def execute(self, input_data: Any) -> Union[Dict[str, Any], None]:
        """Execute the code with the input data."""

        if input_data is None:
            input_data = {}

        if not isinstance(input_data, dict):
            raise TypeError("Input data must be a dictionary.")

        try:
            issues = self.bandit_check(self.code)
            if not issues:
                return {"error": "Security check failed. " + issues}
            from RestrictedPython import compile_restricted
            from RestrictedPython import safe_globals
            from RestrictedPython.Guards import safe_builtins
            from RestrictedPython.Eval import (
                default_guarded_getitem,
                default_guarded_getiter,
            )
            import editdistance
            import textdistance
            from datetime import datetime
            import time
            import textstat

            custom_builtins = safe_builtins.copy()
            custom_builtins.update(
                {
                    "type": type,
                    "dict": dict,
                    "list": list,
                    "set": set,
                    "tuple": tuple,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                    "len": len,
                    "range": range,
                    "enumerate": enumerate,
                    "zip": zip,
                    "sorted": sorted,
                    "min": min,
                    "max": max,
                    "sum": sum,
                    "abs": abs,
                    "all": all,
                    "any": any,
                    "isinstance": isinstance,
                    "issubclass": issubclass,
                    "datetime": datetime,
                    "Exception": Exception,
                    "ValueError": ValueError,
                    "TypeError": TypeError,
                    "KeyError": KeyError,
                    "IndexError": IndexError,
                    "AttributeError": AttributeError,
                    "ImportError": ImportError,
                    "__import__": __import__,
                }
            )

            custom_globals = safe_globals.copy()
            custom_globals.update(
                {
                    "__builtins__": custom_builtins,
                    "json": json,
                    "re": re,
                    "editdistance": editdistance,
                    "textdistance": textdistance,
                    "datetime": datetime,
                    "time": time,
                    "textstat": textstat,
                    "_getitem_": default_guarded_getitem,
                    "_getiter_": default_guarded_getiter,
                    "_write_": lambda x: x,
                }
            )
            # Whitelist of allowed modules
            allowed_modules = {
                "json",
                "re",
                "editdistance",
                "textdistance",
                "datetime",
                "time",
                "textstat",
            }

            def guarded_import(name, *args, **kwargs):
                if name not in allowed_modules:
                    raise ImportError(f"Importing '{name}' is not allowed")
                return __import__(name, *args, **kwargs)

            custom_builtins["__import__"] = guarded_import
            loc = {}
            byte_code = compile_restricted(self.code, "<inline>", "exec")
            exec(byte_code, custom_globals, loc)
            result = loc["main"](**input_data)

            def wrap_non_serializable(obj):
                if isinstance(obj, (str, int, float, list, dict)):
                    if isinstance(obj, list):
                        return [wrap_non_serializable(item) for item in obj]
                    elif isinstance(obj, dict):
                        return {
                            key: wrap_non_serializable(value)
                            for key, value in obj.items()
                        }
                    return obj
                return str(obj)

            wrapped_result = wrap_non_serializable(result)
            return {
                "status": "success",
                "data": wrapped_result,
            }
        except Exception as e:
            return {
                "status": "error",
                "data": f"Failed to execute the code.\nDetails:\n{str(e)}",
            }
