# Step to make an external api call
import json
import time
from typing import Union, Dict, Any, Optional
import aiohttp
from athina.steps.base import Step
import asyncio
from jinja2 import Environment


def prepare_template_data(
    env: Environment,
    template_dict: Optional[Dict[str, str]],
    input_data: Dict[str, Any],
) -> Optional[Dict[str, str]]:
    """Prepare template data by rendering Jinja2 templates."""
    if template_dict is None:
        return None

    prepared_dict = template_dict.copy()
    for key, value in prepared_dict.items():
        prepared_dict[key] = env.from_string(value).render(**input_data)
    return prepared_dict


def debug_json_structure(body_str: str, error: json.JSONDecodeError) -> dict:
    """Analyze JSON structure and identify problematic keys."""
    lines = body_str.split("\n")
    error_line_num = error.lineno - 1

    return {
        "original_body": body_str,
        "problematic_line": (
            lines[error_line_num] if error_line_num < len(lines) else None
        ),
    }


def prepare_body(
    env: Environment, body_template: Optional[str], input_data: Dict[str, Any]
) -> Optional[str]:
    """Prepare request body by rendering Jinja2 template."""
    if body_template is None:
        return None

    return env.from_string(body_template).render(**input_data)


class ApiCall(Step):
    """
    Step that makes an external API call.

    Attributes:
        url: The URL of the API endpoint to call.
        method: The HTTP method to use (e.g., 'GET', 'POST', 'PUT', 'DELETE').
        headers: Optional headers to include in the API request.
        params: Optional params to include in the API request.
        body: Optional request body to include in the API request.
    """

    url: str
    method: str
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, str]] = None
    body: Optional[str] = None
    env: Environment = None
    name: Optional[str] = None
    timeout: int = 30  # Default timeout in seconds
    retries: int = 2  # Default number of retries

    class Config:
        arbitrary_types_allowed = True

    def process_response(
        self,
        status_code: int,
        response_text: str,
        start_time: float,
    ) -> Dict[str, Any]:
        """Process the API response and return formatted result."""
        if status_code >= 400:
            # If the status code is an error, return the error message
            return self._create_step_result(
                status="error",
                data=f"Failed to make the API call.\nStatus code: {status_code}\nError:\n{response_text}",
                start_time=start_time,
            )

        try:
            json_response = json.loads(response_text)
            # If the response is JSON, return the JSON data
            return self._create_step_result(
                status="success",
                data=json_response,
                start_time=start_time,
            )
        except json.JSONDecodeError:
            # If the response is not JSON, return the text
            return self._create_step_result(
                status="success",
                data=response_text,
                start_time=start_time,
            )

    async def execute_async(self, input_data: Any) -> Union[Dict[str, Any], None]:
        """Make an async API call and return the response."""
        start_time = time.perf_counter()

        if input_data is None:
            input_data = {}

        if not isinstance(input_data, dict):
            return self._create_step_result(
                status="error",
                data="Input data must be a dictionary.",
                start_time=start_time,
            )
        # Prepare the environment and input data
        self.env = self._create_jinja_env()

        # Prepare request components
        prepared_body = prepare_body(self.env, self.body, input_data)
        prepared_headers = prepare_template_data(self.env, self.headers, input_data)
        prepared_params = prepare_template_data(self.env, self.params, input_data)
        # Prepare the URL by rendering the template
        prepared_url = self.env.from_string(self.url).render(**input_data)

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        for attempt in range(self.retries):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    try:
                        json_body = (
                            json.loads(prepared_body, strict=False)
                            if prepared_body
                            else None
                        )
                    except json.JSONDecodeError as e:
                        debug_info = debug_json_structure(prepared_body, e)
                        return self._create_step_result(
                            status="error",
                            data=json.dumps(
                                {
                                    "message": f"Failed to parse request body as JSON",
                                    "error_type": "JSONDecodeError",
                                    "error_details": str(e),
                                    "debug_info": debug_info,
                                },
                                indent=2,
                            ),
                            start_time=start_time,
                        )

                    async with session.request(
                        method=self.method,
                        url=prepared_url,
                        headers=prepared_headers,
                        params=prepared_params,
                        json=json_body,
                    ) as response:
                        response_text = await response.text()
                        return self.process_response(
                            response.status, response_text, start_time
                        )

            except asyncio.TimeoutError:
                if attempt < self.retries - 1:
                    await asyncio.sleep(2)
                    continue
                # If the request times out after multiple attempts, return an error message
                return self._create_step_result(
                    status="error",
                    data="Failed to make the API call.\nRequest timed out after multiple attempts.",
                    start_time=start_time,
                )
            except Exception as e:
                # If an exception occurs, return the error message
                return self._create_step_result(
                    status="error",
                    data=f"Failed to make the API call.\nError: {e.__class__.__name__}\nDetails:\n{str(e)}",
                    start_time=start_time,
                )

    def execute(self, input_data: Any) -> Union[Dict[str, Any], None]:
        """Synchronous execute api call that runs the async method in an event loop."""
        return asyncio.run(self.execute_async(input_data))
