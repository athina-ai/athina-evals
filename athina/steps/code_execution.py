from typing import Union, Dict, Any, Optional
from athina.steps import Step
import subprocess
import os
import json
import re
import tempfile
import time


class CodeExecution(Step):
    """
    Step that executes code using the code provided to the step.

    Attributes:
        code: The code to execute.
    """

    code: str
    name: Optional[str] = None

    def format_bandit_result(self, stdout: str) -> str:
        """
        Format the Bandit output into a more readable string.
        """
        try:
            data = json.loads(stdout)
            output = []
            # Add header
            output.append("Security Check Results")
            output.append("=" * 20)
            # Add results
            if data["results"]:
                for result in data["results"]:
                    output.append(f"\nIssue Found:")
                    output.append(f"  Severity: {result['issue_severity']}")
                    output.append(f"  Confidence: {result['issue_confidence']}")
                    output.append(f"  Description: {result['issue_text']}")
                    output.append("\n  Problematic Code:")
                    output.append("  " + "-" * 16)
                    for line in result["code"].splitlines():
                        output.append(f"    {line}")

                    if "issue_cwe" in result:
                        output.append(f"\n  CWE: {result['issue_cwe']['id']}")
                        output.append(f"  CWE Link: {result['issue_cwe']['link']}")

                    output.append(f"  More Info: {result['more_info']}")
            else:
                output.append("\nNo security issues found.")
            # Add metrics summary
            output.append("\nMetrics Summary")
            output.append("-" * 15)
            metrics = data["metrics"]["_totals"]
            output.append(f"Total lines of code: {metrics['loc']}")
            output.append(f"High severity issues: {metrics['SEVERITY.HIGH']}")
            output.append(f"Medium severity issues: {metrics['SEVERITY.MEDIUM']}")
            output.append(f"Low severity issues: {metrics['SEVERITY.LOW']}")
            return "\n".join(output)

        except json.JSONDecodeError:
            return f"Error parsing Bandit output: {stdout}"
        except KeyError as e:
            return f"Error processing Bandit output: Missing key {e}"
        except Exception as e:
            return f"Error processing Bandit output: {e}"

    def bandit_check(self, code: str) -> Optional[str]:
        """
        Run Bandit security check on the provided code.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
            temp_file.write(code.encode("utf-8"))
            temp_file_path = temp_file.name
        try:
            result = subprocess.run(
                ["bandit", "-r", temp_file_path, "-f", "json"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return self.format_bandit_result(result.stdout)
        except Exception as e:
            return str(e)
        finally:
            os.remove(temp_file_path)
        return None

    def execute(self, input_data: Any) -> Union[Dict[str, Any], None]:
        """Execute the code with the input data."""
        start_time = time.perf_counter()

        if input_data is None:
            input_data = {}

        if not isinstance(input_data, dict):
            return self._create_step_result(
                status="error",
                data="Input data must be a dictionary.",
                start_time=start_time,
            )

        try:
            issues = self.bandit_check(self.code)
            if issues:
                return self._create_step_result(
                    status="error",
                    data="Security check failed. Issues:\n" + issues,
                    start_time=start_time,
                )
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
            import textstat
            import urllib

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
                    "urllib": urllib,
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
                "urllib",
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
                if isinstance(obj, (str, bool, int, float, list, dict)):
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
            return self._create_step_result(
                status="success",
                data=wrapped_result,
                start_time=start_time,
            )
        except Exception as e:
            return self._create_step_result(
                status="error",
                data=f"Failed to execute the code.\nDetails:\n{str(e)}",
                start_time=start_time,
            )
