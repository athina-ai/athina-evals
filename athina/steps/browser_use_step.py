import os
from typing import Any, Dict, Optional, List, Union
import asyncio
from dotenv import load_dotenv
import json
import time
from athina.steps import Step
from athina.steps.base import StepResult
from browser_use import Agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr

load_dotenv()


class BrowserUseStep(Step):
    """
    Step that uses browser automation to perform web interactions.

    Attributes:
        openai_api_key: OpenAI API key for the LLM
        model: The OpenAI model to use (default: gpt-4o)
        max_retries: Maximum number of retries for browser actions
        timeout: Timeout in seconds for browser actions
        headless: Whether to run browser in headless mode

    USAGE:
    response: str = BrowserUseStep(model="gpt-4o").execute(input_data={
        "task": "Search for the latest news on the stock market and compile a list of the top 5 most important events."
    })['data']
    """

    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    model: str = "gpt-4o"
    max_retries: int = 3
    timeout: int = 30
    headless: bool = True

    def execute(self, input_data: Any) -> StepResult:
        """Execute browser automation tasks based on user prompts."""
        start_time = time.perf_counter()

        # Ensure input_data is properly formatted
        if isinstance(input_data, dict):
            user_prompts = input_data.get("task", [])
            if isinstance(user_prompts, str):
                user_prompts = [user_prompts]
        elif isinstance(input_data, str):
            user_prompts = [input_data]
        elif isinstance(input_data, list):
            user_prompts = input_data
        else:
            return self._create_step_result(
                status="error",
                data="Input data must be a string, list of strings, or dictionary with 'user_prompts' key",
                start_time=start_time,
            )

        try:
            # Initialize ChatOpenAI with SecretStr
            llm = ChatOpenAI(
                api_key=SecretStr(self.openai_api_key),
                model=self.model,
            )

            # Process each prompt
            results = []
            for prompt in user_prompts:
                # Create and run browser agent
                result = asyncio.run(self._run_browser_agent(llm, prompt))
                results.append({"prompt": prompt, "result": result})

            return self._create_step_result(
                status="success",
                data=json.dumps(results),  # Convert list to JSON string
                start_time=start_time,
            )

        except Exception as e:
            return self._create_step_result(
                status="error",
                data=f"Browser automation failed: {str(e)}",
                start_time=start_time,
            )

    async def _run_browser_agent(self, llm: ChatOpenAI, task: str) -> str:
        """Run a browser agent for a specific task."""
        retry_count = 0
        last_error = None

        while retry_count < self.max_retries:
            try:
                agent = Agent(
                    task=task,
                    llm=llm,
                )
                result = await agent.run()
                return str(result)  # Convert result to string

            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < self.max_retries:
                    # Wait before retrying (exponential backoff)
                    await asyncio.sleep(2**retry_count)

        raise Exception(
            f"Failed after {self.max_retries} retries. Last error: {str(last_error)}"
        )
