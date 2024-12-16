# Step to make an external api call
import json
import time
from typing import Union, Dict, List, Any, Iterable, Optional
import requests
from athina.steps import Step
from jinja2 import Environment
from athina.helpers.jinja_helper import PreserveUndefined


def prepare_input_data(data):
    return {key: json.dumps(value) if isinstance(value, (list, dict)) else value
        for key, value in data.items()}


class SpiderCrawl(Step):
    """
    Step that makes a crawl API Call to https://api.spider.cloud/crawl.

    Attributes:
        url: The query string.
        limit: The maximum amount of pages allowed to crawl per website. Remove the value or set it to 0 to crawl all pages. Defaults to 0.
        metadata: Collect metadata about the content found like page title, description, keywards and etc. This could help improve AI interoperability. Defaults to false.
        return_format: The format of the response. Defaults to raw.
        spider_key: The API key to use for the request.
    """

    url: str
    limit: Optional[int] = 1
    metadata: Optional[bool] = False
    return_format: Optional[str] = "markdown"
    spider_key: str
    env: Environment = None

    class Config:
        arbitrary_types_allowed = True
       

    def execute(self, input_data: Any) -> Union[Dict[str, Any], None]:
        """Make an Search API call and return the response."""

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

        body={
            "url": self.url,
            "limit": self.limit,
            "metadata": self.metadata,
            "return_format": self.return_format
        }
        prepared_body = None
        # Add a filter to the Jinja2 environment to convert the input data to JSON
        body_template = self.env.from_string(json.dumps(body))
        prepared_input_data = prepare_input_data(input_data)
        prepared_body = body_template.render(**prepared_input_data)
        

        retries = 2  # number of retries
        timeout = 300  # seconds
        for attempt in range(retries):
            try:
                response = requests.post(
                    url= "https://api.spider.cloud/crawl",
                    headers={
                        "Content-Type": "application/json",
                      'Authorization': f'Bearer {self.spider_key}',
                    },
                    json=json.loads(prepared_body, strict=False) if prepared_body else None,
                    timeout=timeout,
                )
                if response.status_code >= 400:
                    # If the status code is an error, return the error message
                    return {
                        "status": "error",
                        "data": f"Failed to make the API call.\nStatus code: {response.status_code}\nError:\n{response.text}",
                    }
                try:
                    json_response = response.json()
                    # If the response is JSON, return the JSON data

                    # Loop through the json response and get the content
                    content = []
                    for item in json_response:
                        value = {
                            "content": item.get('content'),
                            "url": item.get('url'),
                            "error": item.get('error'),
                        }
                        content.append(value)

                    
                    return {
                        "status": "success",
                        "data": content,
                    }
                
                except json.JSONDecodeError:
                    # If the response is not JSON, return the text
                    return {
                        "status": "success",
                        "data": response.text,
                    }
            except requests.Timeout:
                if attempt < retries - 1:
                    time.sleep(2)
                    continue
                # If the request times out after multiple attempts, return an error message
                return {
                    "status": "error",
                    "data": "Failed to make the API call.\nRequest timed out after multiple attempts.",
                }
            except Exception as e:
                # If an exception occurs, return the error message
                return {
                    "status": "error",
                    "data": f"Failed to make the API call.\nError: {e.__class__.__name__}\nDetails:\n{str(e)}",
                }
