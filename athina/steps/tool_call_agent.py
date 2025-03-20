import logging
from typing import Any, Dict, Union, Optional, List, AsyncGenerator, Literal, TypedDict
import time
import json
from athina.steps.base import Step
from athina.steps.base import StepResult
import os
import dotenv
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import BaseTool
from datetime import datetime

# Set up logger
logger = logging.getLogger(__name__)


class StreamLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.logs = []

    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)

    def get_logs(self):
        logs = self.logs.copy()
        self.logs = []
        return logs


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
    stream_log_handler: Optional[StreamLogHandler] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        # Set up logging for streaming
        self.stream_log_handler = StreamLogHandler()
        logger.setLevel(logging.INFO)
        logger.addHandler(self.stream_log_handler)

    def _create_step_result(
        self,
        status: Literal["success", "error"],
        data: Any,
        start_time: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StepResult:
        """Create a standardized step result object."""
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        return {
            "status": status,
            "data": str(data),  # Ensure data is a string as required by StepResult
            "metadata": metadata or {},
        }

    def execute(self, input_data: Any) -> StepResult:
        # Import here to avoid issues during initialization
        try:
            from composio_llamaindex import Action, ComposioToolSet
        except ImportError:
            logger.error(
                "composio_llamaindex package not found. Please install it to use this agent."
            )
            return self._create_step_result(
                status="error",
                data="Missing dependency: composio_llamaindex package not found",
                metadata={},
                start_time=time.perf_counter(),
            )

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
                logger.error("No actions specified for the tool call agent")
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
                logger.error("No prompt specified for the tool call agent")
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
            logger.info(f"Tool Call Agent: Initializing with model {self.llm_model}")

            # Initialize the LLM
            llm = OpenAI(model=self.llm_model)

            # Initialize the ComposioToolSet with entity_id if provided
            logger.info(
                f"Tool Call Agent: Setting up ComposioToolSet for entity_id: {entity_id}"
            )
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
                    logger.info(f"Tool Call Agent: Added action {action_name}")
                except AttributeError:
                    # If the action doesn't exist in the enum, log it and continue
                    logger.warning(
                        f"Action '{action_name}' not found in Composio Action enum"
                    )

            # Get the tools from Composio
            all_tools: List[BaseTool] = []
            if action_enums:
                logger.info(
                    f"Tool Call Agent: Getting tools for {len(action_enums)} actions"
                )
                composio_tools = composio_toolset.get_actions(actions=action_enums)
                all_tools.extend(composio_tools)
                logger.info(f"Tool Call Agent: Retrieved {len(all_tools)} tools")

            if not all_tools:
                logger.error("Failed to get any valid tools from Composio")
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
            logger.info("Tool Call Agent: Creating OpenAI agent with tools")
            agent = OpenAIAgent.from_tools(
                tools=all_tools,
                llm=llm,
                prefix_messages=prefix_messages,
                max_function_calls=self.max_function_calls,
                allow_parallel_tool_calls=self.allow_parallel_tool_calls,
                verbose=True,
            )

            # Execute the agent with the prompt
            logger.info("Tool Call Agent: Executing agent with prompt")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Tools: {[tool.metadata.name for tool in all_tools]}")
            logger.info(f"Max function calls: {self.max_function_calls}")
            logger.info(f"Allow parallel tool calls: {self.allow_parallel_tool_calls}")

            response = agent.chat(prompt)

            logger.info("Tool Call Agent: Agent execution completed")

            # Extract tool calls if available
            tool_calls = None
            if hasattr(agent, "get_tool_calls"):
                tool_calls = agent.get_tool_calls()
                logger.info(
                    f"Tool Call Agent: Made {len(tool_calls) if tool_calls else 0} tool calls"
                )

            return self._create_step_result(
                status="success",
                data=str(response),
                metadata={
                    "agent_name": self.agent_name,
                    "entity_id": entity_id,
                    "actions_used": actions,
                    "input_received": input_data,
                    "llm_model": self.llm_model,
                    "tool_calls": tool_calls,
                    "execution_time": time.perf_counter() - start_time,
                },
                start_time=start_time,
            )

        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            logger.error(f"Tool call agent execution failed: {str(e)}")
            logger.error(tb)

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

    async def execute_stream(self, input_data: Any) -> AsyncGenerator[str, None]:
        """Execute the tool call agent with streaming output."""
        start_time = time.perf_counter()

        # Helper function to safely get logs
        def get_logs():
            if self.stream_log_handler:
                return self.stream_log_handler.get_logs()
            return []

        # Validate input
        if not isinstance(input_data, (str, dict)):
            yield json.dumps(
                {
                    "status": "error",
                    "data": "Input must be a string (prompt) or a dictionary with configuration",
                    "metadata": {
                        "logs": get_logs(),
                    },
                }
            )
            return

        try:
            # Import here to avoid issues during initialization
            try:
                from composio_llamaindex import Action, ComposioToolSet
            except ImportError:
                logger.error(
                    "composio_llamaindex package not found. Please install it to use this agent."
                )
                yield json.dumps(
                    {
                        "status": "error",
                        "data": "Missing dependency: composio_llamaindex package not found",
                        "metadata": {
                            "logs": get_logs(),
                        },
                    }
                )
                return

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
                logger.error("No actions specified for the tool call agent")
                yield json.dumps(
                    {
                        "status": "error",
                        "data": "No actions specified for the tool call agent",
                        "metadata": {
                            "agent_name": self.agent_name,
                            "entity_id": entity_id,
                            "input_received": input_data,
                            "logs": get_logs(),
                        },
                    }
                )
                return

            if not prompt:
                logger.error("No prompt specified for the tool call agent")
                yield json.dumps(
                    {
                        "status": "error",
                        "data": "No prompt specified for the tool call agent",
                        "metadata": {
                            "agent_name": self.agent_name,
                            "entity_id": entity_id,
                            "input_received": input_data,
                            "logs": get_logs(),
                        },
                    }
                )
                return

            # Send initial status
            yield json.dumps(
                {
                    "status": "in_progress",
                    "data": "Initializing tool call agent...",
                    "logs": get_logs(),
                }
            )

            # Load environment variables if needed
            dotenv.load_dotenv()
            logger.info(f"Tool Call Agent: Initializing with model {self.llm_model}")

            # Initialize the LLM
            llm = OpenAI(model=self.llm_model)

            # Send log update
            yield json.dumps(
                {
                    "status": "in_progress",
                    "data": f"Initialized LLM with model {self.llm_model}",
                    "logs": get_logs(),
                }
            )

            # Initialize the ComposioToolSet with entity_id if provided
            logger.info(
                f"Tool Call Agent: Setting up ComposioToolSet for entity_id: {entity_id}"
            )
            composio_toolset = (
                ComposioToolSet(entity_id=entity_id) if entity_id else ComposioToolSet()
            )

            # Send log update
            yield json.dumps(
                {
                    "status": "in_progress",
                    "data": "Setting up Composio tools...",
                    "logs": get_logs(),
                }
            )

            # Convert string action names to Action enum values
            action_enums = []
            for action_name in actions:
                try:
                    # Try to get the action from the Action enum by name
                    action_enum = getattr(Action, action_name)
                    action_enums.append(action_enum)
                    logger.info(f"Tool Call Agent: Added action {action_name}")

                    # Send log update for each action
                    yield json.dumps(
                        {
                            "status": "in_progress",
                            "data": f"Added action: {action_name}",
                            "logs": get_logs(),
                        }
                    )
                except AttributeError:
                    # If the action doesn't exist in the enum, log it and continue
                    logger.warning(
                        f"Action '{action_name}' not found in Composio Action enum"
                    )

                    yield json.dumps(
                        {
                            "status": "in_progress",
                            "data": f"Warning: Action '{action_name}' not found",
                            "logs": get_logs(),
                        }
                    )

            # Get the tools from Composio
            all_tools: List[BaseTool] = []
            if action_enums:
                logger.info(
                    f"Tool Call Agent: Getting tools for {len(action_enums)} actions"
                )
                yield json.dumps(
                    {
                        "status": "in_progress",
                        "data": f"Retrieving tools for {len(action_enums)} actions...",
                        "logs": get_logs(),
                    }
                )

                composio_tools = composio_toolset.get_actions(actions=action_enums)
                all_tools.extend(composio_tools)
                logger.info(f"Tool Call Agent: Retrieved {len(all_tools)} tools")

                yield json.dumps(
                    {
                        "status": "in_progress",
                        "data": f"Retrieved {len(all_tools)} tools",
                        "logs": get_logs(),
                    }
                )

            if not all_tools:
                logger.error("Failed to get any valid tools from Composio")
                yield json.dumps(
                    {
                        "status": "error",
                        "data": "Failed to get any valid tools from Composio",
                        "metadata": {
                            "agent_name": self.agent_name,
                            "entity_id": entity_id,
                            "actions_requested": actions,
                            "input_received": input_data,
                            "logs": get_logs(),
                        },
                    }
                )
                return

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
            logger.info("Tool Call Agent: Creating OpenAI agent with tools")
            yield json.dumps(
                {
                    "status": "in_progress",
                    "data": "Creating agent with tools...",
                    "logs": get_logs(),
                }
            )

            agent = OpenAIAgent.from_tools(
                tools=all_tools,
                llm=llm,
                prefix_messages=prefix_messages,
                max_function_calls=self.max_function_calls,
                allow_parallel_tool_calls=self.allow_parallel_tool_calls,
                verbose=True,
            )

            # Execute the agent with the prompt
            logger.info("Tool Call Agent: Executing agent with prompt")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Tools: {[tool.metadata.name for tool in all_tools]}")
            logger.info(f"Max function calls: {self.max_function_calls}")
            logger.info(f"Allow parallel tool calls: {self.allow_parallel_tool_calls}")

            yield json.dumps(
                {
                    "status": "in_progress",
                    "data": "Executing agent with prompt...",
                    "logs": get_logs(),
                }
            )

            # Since OpenAI agent doesn't support streaming inherently, we'll send periodic log updates
            response = agent.chat(prompt)

            logger.info("Tool Call Agent: Agent execution completed")

            # Extract tool calls if available
            tool_calls = None
            if hasattr(agent, "get_tool_calls"):
                tool_calls = agent.get_tool_calls()
                logger.info(
                    f"Tool Call Agent: Made {len(tool_calls) if tool_calls else 0} tool calls"
                )

            # Send final result
            yield json.dumps(
                {
                    "status": "success",
                    "data": str(response),
                    "metadata": {
                        "agent_name": self.agent_name,
                        "entity_id": entity_id,
                        "actions_used": actions,
                        "input_received": input_data,
                        "llm_model": self.llm_model,
                        "tool_calls": tool_calls,
                        "logs": get_logs(),
                        "execution_time": time.perf_counter() - start_time,
                    },
                }
            )

        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            logger.error(f"Tool call agent execution failed: {str(e)}")
            logger.error(tb)

            yield json.dumps(
                {
                    "status": "error",
                    "data": f"Tool call agent execution failed: {str(e)}",
                    "metadata": {
                        "agent_name": self.agent_name,
                        "entity_id": entity_id,
                        "traceback": tb,
                        "input_received": input_data,
                        "logs": get_logs(),
                        "execution_time": time.perf_counter() - start_time,
                    },
                }
            )
