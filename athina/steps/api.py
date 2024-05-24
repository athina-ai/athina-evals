# Step to make an external api call
import json
from typing import Union, Dict, Any, Iterable, Optional
import requests
from athina.steps import Step
from jinja2 import Environment
from athina.helpers.jinja_helper import PreserveUndefined


class ApiCall(Step):
    """
    Step that makes an external API call.

    Attributes:
        url: The URL of the API endpoint to call.
        method: The HTTP method to use (e.g., 'GET', 'POST', 'PUT', 'DELETE').
        headers: Optional headers to include in the API request.
        params: Optional query parameters to include in the API request.
        body: Optional request body to include in the API request.
        expected_status_codes: Expected HTTP status codes for a successful response.
    """

    url: str
    method: str
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    body: Optional[str] = None
    expected_status_codes: Iterable[int] = (200,)
    env: Environment = None

    class Config:
        arbitrary_types_allowed = True

    def execute(self, input_data: Any) -> Union[Dict[str, Any], None]:
        """Make an API call and return the response."""

        if input_data is None:
            input_data = {}

        if not isinstance(input_data, dict):
            raise TypeError("Input data must be a dictionary.")

        # Create a custom Jinja2 environment with double curly brace delimiters and PreserveUndefined
        self.env = Environment(
            variable_start_string='{{', 
            variable_end_string='}}',
            undefined=PreserveUndefined
        )

        if self.body is not None:
            body_template = self.env.from_string(self.body)
            self.body = body_template.render(**input_data)

        response = requests.request(
            method=self.method,
            url=self.url,
            headers=self.headers,
            params=self.params,
            json=json.loads(self.body),
        )

        if response.status_code in self.expected_status_codes:
            return response.json()
        else:
            return None
