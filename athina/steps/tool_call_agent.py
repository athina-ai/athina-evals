from typing import Any, Dict, Union, Optional, List
import time
from athina.steps.base import Step
from athina.steps.base import StepResult
import os
import dotenv
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import BaseTool
from datetime import datetime


class ToolCallAgent(Step):
    """
    Step that acts as an agent to handle tool calls using LlamaIndex's OpenAI agent with Composio tools.

    This agent will:
    1. Get the specified Composio tools
    2. Create an OpenAI agent with those tools
    3. Run the agent with the provided prompt
    """

    # Define the required attributes
    agent_name: str
    entity_id: Optional[str] = None
    actions: Optional[List[str]] = None  # List of Composio action names
    prompt: Optional[str] = None  # Prompt to send to the agent
    llm_model: str = "gpt-4o"  # Default model to use
    max_function_calls: int = 15  # Default max function calls
    allow_parallel_tool_calls: bool = False  # Default parallel tool calls setting

    def execute(self, input_data: Any) -> StepResult:
        from composio_llamaindex import Action, ComposioToolSet

        """Execute the tool call agent with LlamaIndex and Composio tools."""
        start_time = time.perf_counter()

        try:
            # Extract entity_id from config or input data
            entity_id = self.entity_id

            # If entity_id is in the input data, use that instead (overrides the config)
            if isinstance(input_data, dict) and "entity_id" in input_data:
                entity_id = input_data["entity_id"]

            # Extract prompt from config or input data
            prompt = self.prompt
            if isinstance(input_data, dict) and "prompt" in input_data:
                prompt = input_data["prompt"]

            # Extract actions from config or input data
            actions = self.actions or []
            if (
                isinstance(input_data, dict)
                and "actions" in input_data
                and input_data["actions"]
            ):
                actions = input_data["actions"]

            if not actions:
                return self._create_step_result(
                    status="error",
                    data="No actions specified for the tool call agent",
                    metadata={
                        "agent_name": self.agent_name,
                        "entity_id": entity_id,
                        "input_received": input_data,
                    },
                    start_time=start_time,
                )

            if not prompt:
                return self._create_step_result(
                    status="error",
                    data="No prompt specified for the tool call agent",
                    metadata={
                        "agent_name": self.agent_name,
                        "entity_id": entity_id,
                        "input_received": input_data,
                    },
                    start_time=start_time,
                )

            # Load environment variables if needed
            dotenv.load_dotenv()

            # Initialize the LLM
            llm = OpenAI(model=self.llm_model)

            # Initialize the ComposioToolSet with entity_id if provided
            composio_toolset = (
                ComposioToolSet(entity_id=entity_id) if entity_id else ComposioToolSet()
            )

            # Convert string action names to Action enum values
            action_enums = []
            for action_name in actions:
                try:
                    # Try to get the action from the Action enum by name
                    action_enum = getattr(Action, action_name)
                    action_enums.append(action_enum)
                except AttributeError:
                    # If the action doesn't exist in the enum, log it and continue
                    print(
                        f"Warning: Action '{action_name}' not found in Composio Action enum"
                    )

            # Get the tools from Composio
            all_tools: List[BaseTool] = []
            if action_enums:
                composio_tools = composio_toolset.get_actions(actions=action_enums)
                all_tools.extend(composio_tools)

            if not all_tools:
                return self._create_step_result(
                    status="error",
                    data="Failed to get any valid tools from Composio",
                    metadata={
                        "agent_name": self.agent_name,
                        "entity_id": entity_id,
                        "actions_requested": actions,
                        "input_received": input_data,
                    },
                    start_time=start_time,
                )

            # Set up system prompt for the agent
            prefix_messages = [
                ChatMessage(
                    role="system",
                    content=(
                        f"You are an assistant named {self.agent_name} that helps users accomplish tasks using various tools. "
                        "Use the provided tools to fulfill the user's request. "
                        f"Today's date is {datetime.now().strftime('%B %d, %Y')}."
                    ),
                )
            ]

            # Create an agent with the tools
            agent = OpenAIAgent.from_tools(
                tools=all_tools,
                llm=llm,
                prefix_messages=prefix_messages,
                max_function_calls=self.max_function_calls,
                allow_parallel_tool_calls=self.allow_parallel_tool_calls,
                verbose=True,
            )

            # Execute the agent with the prompt
            response = agent.chat(prompt)

            return self._create_step_result(
                status="success",
                data=str(response),
                metadata={
                    "agent_name": self.agent_name,
                    "entity_id": entity_id,
                    "actions_used": actions,
                    "input_received": input_data,
                    "llm_model": self.llm_model,
                    "tool_calls": (
                        agent.get_tool_calls()
                        if hasattr(agent, "get_tool_calls")
                        else None
                    ),
                },
                start_time=start_time,
            )

        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            return self._create_step_result(
                status="error",
                data=f"Tool call agent execution failed: {str(e)}",
                metadata={
                    "agent_name": self.agent_name,
                    "entity_id": entity_id,
                    "traceback": tb,
                    "input_received": input_data,
                },
                start_time=start_time,
            )
