import os
import json
import logging
import tiktoken
from typing import Dict, Any, Optional, List, Literal, AsyncGenerator
from athina.steps import Step
from dotenv import load_dotenv
import time
import asyncio
from athina.llms.litellm_service import LitellmService
from jinja2 import Environment

# Configure logging with both file and console handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Create a custom handler that captures logs for streaming
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


# Create console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console_handler)

# Load environment variables
load_dotenv()

# Token limits for different models
MODEL_TOKEN_LIMITS = {
    "gpt-4-turbo-preview": 128000,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4": 8192,
    "gpt-3.5-turbo": 16385,
}

DEFAULT_MODEL = "gpt-4o-mini"


def get_token_count(text: str, model: str = DEFAULT_MODEL) -> int:
    """Count the number of tokens in a text string."""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.debug(f"Error counting tokens: {e}")
        # Fallback to approximate count (1 token ≈ 4 chars)
        return len(text) // 4


def truncate_to_token_limit(
    text: str, max_tokens: int, model: str = DEFAULT_MODEL
) -> str:
    """Truncate text to fit within token limit while preserving sentence boundaries."""
    current_tokens = get_token_count(text, model)

    if current_tokens <= max_tokens:
        return text

    try:
        encoding = tiktoken.encoding_for_model(model)
        tokens = encoding.encode(text)
        truncated = encoding.decode(tokens[:max_tokens])

        # Try to end at a sentence boundary
        last_period = truncated.rfind(".")
        if last_period > 0:
            truncated = truncated[: last_period + 1]

        return truncated
    except Exception as e:
        logger.debug(f"Error truncating text: {e}")
        # Fallback to simple character-based truncation
        ratio = max_tokens / current_tokens
        char_limit = int(len(text) * ratio)
        return text[:char_limit]


def prepare_for_llm(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    max_tokens_for_completion: int = 1000,
) -> List[Dict[str, str]]:
    """Prepare messages for LLM by ensuring they fit within context window."""
    model_limit = MODEL_TOKEN_LIMITS.get(model, 8192)
    available_tokens = model_limit - max_tokens_for_completion

    total_tokens = sum(get_token_count(msg["content"], model) for msg in messages)

    if total_tokens <= available_tokens:
        return messages

    # Keep system message as is, truncate user/assistant messages if needed
    system_messages = [msg for msg in messages if msg["role"] == "system"]
    other_messages = [msg for msg in messages if msg["role"] != "system"]

    system_tokens = sum(
        get_token_count(msg["content"], model) for msg in system_messages
    )
    tokens_per_message = (available_tokens - system_tokens) // len(other_messages)

    truncated_messages = []
    truncated_messages.extend(system_messages)

    for msg in other_messages:
        content = msg["content"]
        if get_token_count(content, model) > tokens_per_message:
            content = truncate_to_token_limit(content, tokens_per_message, model)
        truncated_messages.append({"role": msg["role"], "content": content})

    return truncated_messages


class ResearchAgent(Step):
    """
    Step that performs iterative research using search and LLM capabilities.

    Attributes:
        openai_api_key: OpenAI API key for LLM interactions
        exa_api_key: Exa API key for search operations
        perplexity_api_key: Perplexity API key for search operations
        search_provider: Search provider to use ('exa' or 'perplexity')
        max_iterations: Maximum number of research iterations
        model: LLM model to use
        prompt: The research prompt template with optional Jinja2 variables
    """

    openai_api_key: str
    exa_api_key: str = ""
    perplexity_api_key: str = ""
    search_provider: str = "perplexity"
    max_iterations: int = 3
    model: str = DEFAULT_MODEL
    num_search_queries: int = 10
    prompt: str = ""
    llm_service: Any = None
    research_context: List[Dict[str, Any]] = []
    stream_log_handler: Optional[StreamLogHandler] = None
    env: Optional[Environment] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self.openai_api_key = self.openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.exa_api_key = self.exa_api_key or os.getenv("EXA_API_KEY", "")
        self.perplexity_api_key = self.perplexity_api_key or os.getenv(
            "PERPLEXITY_API_KEY", ""
        )
        self.search_provider = self.search_provider.lower()

        if self.search_provider not in ["exa", "perplexity"]:
            logger.warning(
                f"Invalid search provider '{self.search_provider}'. Defaulting to 'exa'."
            )
            self.search_provider = "exa"

        if self.search_provider == "exa" and not self.exa_api_key:
            logger.warning(
                "Exa API key not provided. Search functionality may not work properly."
            )
        elif self.search_provider == "perplexity" and not self.perplexity_api_key:
            logger.warning(
                "Perplexity API key not provided. Search functionality may not work properly."
            )

        self.llm_service = LitellmService(api_key=self.openai_api_key)
        self.num_search_queries = self.num_search_queries or 10
        self.research_context = []
        self.stream_log_handler = StreamLogHandler()
        self.stream_log_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(self.stream_log_handler)
        logger.info(
            f"Research Agent initialized with {self.max_iterations} iterations and {self.num_search_queries} search queries using model {self.model} and {self.search_provider} search provider"
        )
        self.env = self._create_jinja_env()

    def _create_jinja_env(self) -> Environment:
        """Create a Jinja2 environment for template rendering."""
        return Environment(trim_blocks=True, lstrip_blocks=True, autoescape=False)

    def _create_step_result(
        self,
        status: Literal["success", "error", "in_progress"],
        data: Any,
        start_time: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a standardized result object."""
        execution_time_ms = round((time.time() - start_time) * 1000)

        if metadata is None:
            metadata = {}

        metadata["response_time"] = execution_time_ms

        return {"status": status, "data": data, "metadata": metadata}

    def _extract_evaluation_criteria(self, prompt: str) -> Dict[str, Any]:
        """Extract evaluation criteria and initial search queries from the prompt."""
        try:
            logger.info(
                "🔍 Analyzing research prompt to extract evaluation criteria and search queries..."
            )

            NUM_EVALUATION_STATEMENTS = 5
            system_prompt = f"""You are a research planning assistant. Your task is to analyze a research prompt and create evaluation criteria and search queries.

Return your response in the following JSON format ONLY, with no additional text:
{{
    "evaluation_statements": {{
        "evaluation": [
            {{"statement": "...", "status": "fail"}},
            {{"statement": "...", "status": "fail"}}
        ]
    }},
    "search_queries": [
        "specific search query 1",
        "specific search query 2"
    ]
}}

Evaluation statements are statements that can be used to determine if the research is complete as related to the prompt. 

For example, if the prompt is "Sam Altman", the evaluation statements could be:
"Research includes comprehensive information about Sam Altman background, career, and accomplishments"
"Research includes comprehensive information about Sam Altman's education"
"Research includes comprehensive information about Sam Altman's work experience"
"Research includes comprehensive information about Sam Altman's personal life"
"Research includes comprehensive information about Sam Altman's political views"
"Research includes comprehensive information about Sam Altman's philanthropic work"

For example, if the prompt is "Analyze the market opportunity for a new AI-powered personal assistant", the evaluation statements could be:
"Research includes comprehensive information about the market opportunity for a new AI-powered personal assistant"
"Research includes competitive analysis of existing AI-powered personal assistants"
"Research includes information about the target audience for the new AI-powered personal assistant"
"Research includes information about the key features of the new AI-powered personal assistant"
"Research includes information about the potential revenue for the new AI-powered personal assistant"

Guidelines:
Think carefully about the user's prompt to create appropriate search queries and evaluation statements. 
The search queries are meant to be used to gather information as research for the user's prompt.
The evaluation statements are meant to be used to determine if the research is complete as related to the prompt.

1. Create exactly {NUM_EVALUATION_STATEMENTS} specific evaluation statements that can be used to determine if the research is complete as related to the prompt
2. Create exactly {self.num_search_queries} specific, well-formed search queries that would help gather relevant information.
3. All evaluation statements should initially have "status": "fail"
4. Evaluation statements should be specific and directly related to the prompt. For example, if the prompt is "Sam Altman".
5. Search queries should be specific and directly related to the evaluation statements"""

            response_content = self.llm_service.chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )

            if not response_content:
                raise ValueError("Empty response from LLM")

            result = json.loads(response_content)

            # Log the extracted information
            logger.info("Research Agent: Identified evaluation criteria:")
            for stmt in result.get("evaluation_statements", {}).get("evaluation", []):
                logger.info(
                    f"Research Agent: Criterion - {stmt['statement']} (Initial Status: {stmt['status']})"
                )

            logger.info("Research Agent: Generated initial search queries:")
            for query in result.get("search_queries", []):
                logger.info(f"Research Agent: Query - {query}")

            return result
        except Exception as e:
            logger.error(
                f"Research Agent: Error extracting evaluation criteria: {str(e)}"
            )
            return {
                "evaluation_statements": {
                    "evaluation": [
                        {"statement": "Research is comprehensive", "status": "fail"}
                    ]
                },
                "search_queries": [f"comprehensive information about {prompt}"],
            }

    def _execute_search(self, query: str) -> List[Dict[str, Any]]:
        """Execute a search query using the configured search provider."""
        logger.info(
            f"Research Agent: Executing search with {self.search_provider}: '{query}'"
        )

        if self.search_provider == "perplexity":
            return self._execute_perplexity_search(query)
        else:
            return self._execute_exa_search(query)

    def _execute_exa_search(self, query: str) -> List[Dict[str, Any]]:
        """Execute a search query using Exa search API."""
        try:
            import requests

            url = "https://api.exa.ai/search"
            headers = {
                "content-type": "application/json",
                "Authorization": f"Bearer {self.exa_api_key}",
            }
            payload = {"query": query, "contents": {"text": True}}

            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()

            results = response.json()
            if results and isinstance(results, dict) and results.get("results"):
                data = results.get("results", [])
                logger.info(
                    f"Research Agent: Retrieved {len(data)} results from Exa search"
                )
                return data

            logger.warning(
                f"Research Agent: Exa search returned invalid results format"
            )
            return []

        except Exception as e:
            logger.error(f"Research Agent: Exa search error: {str(e)}")
            return []

    def _execute_perplexity_search(self, query: str) -> List[Dict[str, Any]]:
        """Execute a search query using Perplexity Sonar API."""
        try:
            import requests

            url = "https://api.perplexity.ai/chat/completions"
            headers = {
                "content-type": "application/json",
                "Authorization": f"Bearer {self.perplexity_api_key}",
            }
            payload = {
                "model": "sonar",
                "messages": [
                    {
                        "role": "system",
                        "content": "Be precise and concise. Provide factual information with citations.",
                    },
                    {"role": "user", "content": query},
                ],
                "temperature": 0.2,
                "top_p": 0.9,
                "return_images": False,
                "return_related_questions": False,
                "stream": False,
            }

            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()

            results = response.json()

            # Transform Perplexity response to match Exa format for compatibility
            transformed_results = []

            if results and isinstance(results, dict):
                # Extract content from the first choice
                choices = results.get("choices", [])
                if choices and len(choices) > 0:
                    content = choices[0].get("message", {}).get("content", "")

                    # Get citations
                    citations = results.get("citations", [])

                    # Create a single result with the content
                    transformed_results.append(
                        {
                            "text": content,
                            "url": "perplexity_search_result",
                            "title": "Perplexity Search Result",
                        }
                    )

                    # Add each citation as a separate result
                    for i, citation in enumerate(citations):
                        transformed_results.append(
                            {
                                "text": f"Citation {i+1}",
                                "url": citation,
                                "title": f"Citation {i+1}",
                            }
                        )

                    logger.info(
                        f"Research Agent: Retrieved Perplexity search result with {len(citations)} supporting citations"
                    )
                    return transformed_results

            logger.warning(
                f"Research Agent: Perplexity search returned invalid results format"
            )
            return []

        except Exception as e:
            logger.error(f"Research Agent: Perplexity search error: {str(e)}")
            return []

    def _evaluate_progress(
        self, context: str, evaluation_statements: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Evaluate research progress against the evaluation statements."""
        try:
            logger.info("Research Agent: Evaluating research progress against criteria")

            system_prompt = """Given the current research context and evaluation statements, determine which criteria have been met.
For each statement, mark it as "pass" if the criteria has been satisfied based on the context.
Return the updated evaluation statements as a JSON array.
The JSON array should be in the following format:
{
    "evaluation": [
        {"statement": "...", "status": "pass"},
        {"statement": "...", "status": "fail"}
    ]
}"""
            response_content = self.llm_service.chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Context: {context}\nEvaluation statements: {json.dumps(evaluation_statements)}",
                    },
                ],
                response_format={"type": "json_object"},
            )

            if not response_content:
                raise ValueError("Empty response from LLM")

            updated_statements = json.loads(response_content)
            logger.info(f"Research Agent: Updated evaluation criteria status")
            for stmt in updated_statements:
                if isinstance(stmt, dict):
                    logger.info(
                        f"Research Agent: Criterion '{stmt.get('statement')}' - Status: {stmt.get('status')}, Reason: {stmt.get('reason', 'No reason provided')}"
                    )
                else:
                    logger.warning(
                        f"Research Agent: Invalid evaluation statement format: {stmt}"
                    )

            return {"evaluation": updated_statements}
        except Exception as e:
            logger.error(
                f"Research Agent: Error evaluating research progress: {str(e)}"
            )
            return evaluation_statements

    def _synthesize_findings(self, prompt: str, context: str) -> str:
        """Synthesize research findings into a coherent response."""
        try:
            logger.info(
                "Research Agent: Synthesizing research findings into a coherent response"
            )

            # Calculate available tokens for context
            # Reserve tokens for the system prompt, user prompt, and response
            SYSTEM_PROMPT_TOKENS = 500  # Approximate tokens for system prompt
            USER_PROMPT_TOKENS = 100  # Approximate tokens for user prompt
            RESPONSE_TOKENS = 2000  # Reserve tokens for response
            model_limit = MODEL_TOKEN_LIMITS.get(self.model, 8192)
            available_context_tokens = model_limit - (
                SYSTEM_PROMPT_TOKENS + USER_PROMPT_TOKENS + RESPONSE_TOKENS
            )

            # Truncate context if needed
            if get_token_count(context, self.model) > available_context_tokens:
                logger.info(
                    f"⚠️ Context exceeds token limit. Truncating to {available_context_tokens} tokens..."
                )
                context = truncate_to_token_limit(
                    context, available_context_tokens, self.model
                )

            system_prompt = """Given the user prompt and accumulated context, synthesize a comprehensive, college-level report about the prompt.

Your response must follow these requirements:

Structure and Formatting:
1. Begin with a clear executive summary or introduction that is clearly related to the prompt
2. Use clear hierarchical headings and subheadings to organize content in a way that is easy to read and related to the prompt
3. Break complex information into digestible sections
4. End with a concise conclusion or key takeaways

Content Quality and Citations:
1. Write at a college academic level (clear, precise, and sophisticated language)
2. Include inline citations for EVERY claim or piece of information using markdown links
   - Format: "According to [this research](source_url), the finding shows..."
   - Every paragraph must have at least one citation
   - Link directly to the source URL in the markdown citation
3. Synthesize information from multiple sources rather than just summarizing
4. Present balanced viewpoints when addressing controversial topics
5. Include quantitative data and specific examples where relevant
6. Do NOT make up any information. ONLY use the information provided in the research context.

Readability:
1. Use professional but accessible language (avoid jargon unless necessary)
2. Employ topic sentences to guide readers through your arguments
3. Create logical transitions between sections
4. Use bullet points or numbered lists for complex enumerations
5. Maintain consistent formatting throughout the document

Citation Requirements:
1. Every major claim must have an inline markdown citation
2. Citations must be seamlessly integrated into the text flow
3. Use the exact source URLs provided in the research context
4. Multiple citations in a single sentence should be separated by semicolons

The final report should demonstrate thorough research, critical analysis, and clear communication while remaining directly relevant to the user's prompt.
"""

            response_content = self.llm_service.chat_completion(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"### PROMPT\n{prompt}\n\n### CONTEXT\n{context}",
                    },
                ],
            )

            if not response_content:
                return "Error: No response from LLM"

            logger.info(
                f"Research Agent: Completed synthesis of research findings ({get_token_count(response_content, self.model)} tokens)"
            )
            return response_content
        except Exception as e:
            logger.error(f"Research Agent: Error synthesizing findings: {str(e)}")
            return "Error synthesizing research findings."

    def execute(self, input_data: Any) -> Dict[str, Any]:
        """Execute the research process without streaming."""
        start_time = time.time()

        # Validate input
        if not isinstance(input_data, dict):
            return self._create_step_result(
                status="error",
                data="Input must be a dictionary for variable interpolation",
                start_time=start_time,
            )

        try:
            # Ensure env is initialized
            if self.env is None:
                self.env = self._create_jinja_env()

            # Interpolate the prompt with variables from input_data
            try:
                resolved_prompt = self.env.from_string(self.prompt).render(**input_data)
            except Exception as e:
                return self._create_step_result(
                    status="error",
                    data=f"Error interpolating prompt template: {str(e)}",
                    start_time=start_time,
                )

            if not resolved_prompt:
                return self._create_step_result(
                    status="error",
                    data="No research prompt provided or empty prompt after interpolation",
                    start_time=start_time,
                )

            logger.info(f"🔍 Starting research on: {resolved_prompt}")

            # Extract evaluation criteria and initial queries
            eval_result = self._extract_evaluation_criteria(resolved_prompt)
            evaluation_statements = eval_result.get(
                "evaluation_statements", {"evaluation": []}
            )
            search_queries = eval_result.get("search_queries", [])

            # Initialize research context
            self.research_context = []
            sources = []

            # Execute initial searches
            for query in search_queries:
                results = self._execute_search(query)
                for result in results:
                    source = str(result.get("url", ""))
                    content = str(result.get("text", ""))

                    # Skip empty results
                    if not content:
                        continue

                    if source and source not in sources:
                        sources.append(source)

                    # For Perplexity, the first result contains the main content
                    if (
                        self.search_provider == "perplexity"
                        and source == "perplexity_search_result"
                    ):
                        result_type = "perplexity_answer"
                    else:
                        result_type = "search"

                    self.research_context.append(
                        {
                            "type": result_type,
                            "query": query,
                            "content": content,
                            "source": source,
                        }
                    )

            # Main research loop
            iteration = 0
            while iteration < self.max_iterations:
                # Combine context for evaluation
                current_context = "\n".join(
                    [
                        f"{item['type']} - {item['source']} - {item['content']}"
                        for item in self.research_context
                    ]
                )

                # Truncate if needed
                max_context_tokens = MODEL_TOKEN_LIMITS.get(self.model, 8192) - 1000
                if get_token_count(current_context, self.model) > max_context_tokens:
                    current_context = truncate_to_token_limit(
                        current_context, max_context_tokens, self.model
                    )

                # Evaluate progress
                evaluation_statements = self._evaluate_progress(
                    current_context, evaluation_statements
                )

                # Generate next search query if needed
                if iteration < self.max_iterations - 1:
                    next_query_prompt = f"""Based on the current research progress, the user prompt, and evaluation statements, what should be the next search query? Return only the search query text. Consider the prompt carefully - we should search for information related to the prompt."""

                    response_content = self.llm_service.chat_completion(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": next_query_prompt},
                            {
                                "role": "user",
                                "content": f"Context: {current_context}\nPrompt: {resolved_prompt}\nEvaluation statements: {json.dumps(evaluation_statements)}",
                            },
                        ],
                    )

                    if not response_content:
                        logger.warning("Empty response when generating next query")
                        continue

                    next_query = response_content.strip()
                    logger.info(f"🔍 Following up on: {next_query}")

                    # Execute the follow-up search
                    results = self._execute_search(next_query)
                    for result in results:
                        source = str(result.get("url", ""))
                        if source and source not in sources:
                            sources.append(source)

                        self.research_context.append(
                            {
                                "type": "search",
                                "query": next_query,
                                "content": str(result.get("text", "")),
                                "source": source,
                            }
                        )

                iteration += 1

            if iteration >= self.max_iterations:
                logger.info("⚠️  Reached research depth limit")

            # Synthesize findings
            final_context = "\n".join(
                [
                    f"{item['type']} - {item['source']} - {item['content']}"
                    for item in self.research_context
                ]
            )
            synthesis = self._synthesize_findings(resolved_prompt, final_context)

            logger.info("✅ Research complete!")

            # Get all logs for the synchronous execution
            logs = []
            if self.stream_log_handler:
                logs = self.stream_log_handler.get_logs()

            return self._create_step_result(
                status="success",
                data=synthesis,
                start_time=start_time,
                metadata={
                    "logs": logs,
                    "evaluation_statements": (
                        evaluation_statements["evaluation"]
                        if isinstance(evaluation_statements, dict)
                        and "evaluation" in evaluation_statements
                        else []
                    ),
                    "sources": sources,
                    "iterations": iteration + 1,
                    "total_sources": len(sources),
                    "criteria_met": isinstance(evaluation_statements, dict)
                    and "evaluation" in evaluation_statements
                    and all(
                        isinstance(stmt, dict) and stmt.get("status", "") == "pass"
                        for stmt in evaluation_statements["evaluation"]
                    ),
                    "stage": "complete",
                },
            )

        except Exception as e:
            logger.error(f"Error during research: {str(e)}")
            logs = []
            if self.stream_log_handler:
                logs = self.stream_log_handler.get_logs()

            return self._create_step_result(
                status="error",
                data=f"Research process failed: {str(e)}",
                start_time=start_time,
                metadata={
                    "logs": logs,
                },
            )

    async def execute_stream(self, input_data: Any) -> AsyncGenerator[str, None]:
        """Execute the research process with streaming output."""
        start_time = time.time()

        # Helper function to safely get logs
        def get_logs():
            if self.stream_log_handler:
                return self.stream_log_handler.get_logs()
            return []

        # Validate input
        if not isinstance(input_data, dict):
            yield json.dumps(
                self._create_step_result(
                    status="error",
                    data="Input must be a dictionary for variable interpolation",
                    start_time=start_time,
                )
            )
            return

        try:
            # Ensure env is initialized
            if self.env is None:
                self.env = self._create_jinja_env()

            # Interpolate the prompt with variables from input_data
            try:
                resolved_prompt = self.env.from_string(self.prompt).render(**input_data)
            except Exception as e:
                yield json.dumps(
                    self._create_step_result(
                        status="error",
                        data=f"Error interpolating prompt template: {str(e)}",
                        start_time=start_time,
                    )
                )
                return

            if not resolved_prompt:
                yield json.dumps(
                    self._create_step_result(
                        status="error",
                        data="No research prompt provided or empty prompt after interpolation",
                        start_time=start_time,
                    )
                )
                return

            logger.info(f"🔍 Starting research on: {resolved_prompt}")
            yield json.dumps(
                self._create_step_result(
                    status="in_progress",
                    data="",
                    start_time=start_time,
                    metadata={"logs": get_logs(), "stage": "initialization"},
                )
            )

            # Extract evaluation criteria and initial queries
            eval_result = self._extract_evaluation_criteria(resolved_prompt)
            evaluation_statements = eval_result.get(
                "evaluation_statements", {"evaluation": []}
            )
            search_queries = eval_result.get("search_queries", [])

            yield json.dumps(
                self._create_step_result(
                    status="in_progress",
                    data="",
                    start_time=start_time,
                    metadata={
                        "logs": get_logs(),
                        "evaluation_statements": evaluation_statements,
                        "search_queries": search_queries,
                        "stage": "planning",
                    },
                )
            )

            # Initialize research context
            self.research_context = []
            sources = []

            # Execute initial searches
            for i, query in enumerate(search_queries):
                logger.info(
                    f"🔍 Executing search {i+1}/{len(search_queries)}: '{query}'"
                )

                yield json.dumps(
                    self._create_step_result(
                        status="in_progress",
                        data="",
                        start_time=start_time,
                        metadata={
                            "logs": get_logs(),
                            "current_query": query,
                            "stage": "initial_search",
                            "search_progress": f"{i+1}/{len(search_queries)}",
                        },
                    )
                )

                results = self._execute_search(query)
                for result in results:
                    source = str(result.get("url", ""))
                    if source and source not in sources:
                        sources.append(source)

                    self.research_context.append(
                        {
                            "type": "search",
                            "query": query,
                            "content": str(result.get("text", "")),
                            "source": source,
                        }
                    )

                await asyncio.sleep(0.1)  # Small delay to avoid overwhelming the client

                yield json.dumps(
                    self._create_step_result(
                        status="in_progress",
                        data="",
                        start_time=start_time,
                        metadata={
                            "logs": get_logs(),
                            "sources": sources,
                            "stage": "search_completed",
                            "search_progress": f"{i+1}/{len(search_queries)}",
                        },
                    )
                )

            # Main research loop
            iteration = 0
            while iteration < self.max_iterations:
                logger.info(
                    f"📚 Research iteration {iteration+1}/{self.max_iterations}"
                )

                # Combine context for evaluation
                current_context = "\n".join(
                    [
                        f"{item['type']} - {item['source']} - {item['content']}"
                        for item in self.research_context
                    ]
                )

                # Truncate if needed
                max_context_tokens = MODEL_TOKEN_LIMITS.get(self.model, 8192) - 1000
                if get_token_count(current_context, self.model) > max_context_tokens:
                    current_context = truncate_to_token_limit(
                        current_context, max_context_tokens, self.model
                    )

                # Evaluate progress
                logger.info("📊 Evaluating research progress...")
                yield json.dumps(
                    self._create_step_result(
                        status="in_progress",
                        data="",
                        start_time=start_time,
                        metadata={
                            "logs": get_logs(),
                            "stage": "evaluating",
                            "iteration": f"{iteration+1}/{self.max_iterations}",
                            "sources": sources,
                        },
                    )
                )

                evaluation_statements = self._evaluate_progress(
                    current_context, evaluation_statements
                )

                yield json.dumps(
                    self._create_step_result(
                        status="in_progress",
                        data="",
                        start_time=start_time,
                        metadata={
                            "logs": get_logs(),
                            "evaluation_statements": (
                                evaluation_statements["evaluation"]
                                if isinstance(evaluation_statements, dict)
                                and "evaluation" in evaluation_statements
                                else []
                            ),
                            "stage": "evaluation_complete",
                            "iteration": f"{iteration+1}/{self.max_iterations}",
                            "sources": sources,
                        },
                    )
                )

                # Check if all criteria are met
                if (
                    isinstance(evaluation_statements, dict)
                    and "evaluation" in evaluation_statements
                    and all(
                        isinstance(stmt, dict) and stmt.get("status", "") == "pass"
                        for stmt in evaluation_statements["evaluation"]
                    )
                ):
                    logger.info("✨ Research criteria satisfied!")
                    break

                # Generate next search query if needed
                if iteration < self.max_iterations - 1:
                    logger.info("🔍 Generating follow-up search query...")
                    yield json.dumps(
                        self._create_step_result(
                            status="in_progress",
                            data="",
                            start_time=start_time,
                            metadata={
                                "logs": get_logs(),
                                "stage": "generating_query",
                                "iteration": f"{iteration+1}/{self.max_iterations}",
                                "sources": sources,
                            },
                        )
                    )

                    next_query_prompt = f"""Based on the current research progress and evaluation statements, what should be the next search query? Return only the search query text."""

                    response_content = self.llm_service.chat_completion(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": next_query_prompt},
                            {
                                "role": "user",
                                "content": f"Context: {current_context}\nEvaluation statements: {json.dumps(evaluation_statements)}\nPrompt: {resolved_prompt}",
                            },
                        ],
                    )

                    if not response_content:
                        logger.warning("Empty response when generating next query")
                        continue

                    next_query = response_content.strip()
                    logger.info(f"🔍 Following up on: {next_query}")

                    yield json.dumps(
                        self._create_step_result(
                            status="in_progress",
                            data="",
                            start_time=start_time,
                            metadata={
                                "logs": get_logs(),
                                "stage": "executing_followup",
                                "current_query": next_query,
                                "iteration": f"{iteration+1}/{self.max_iterations}",
                                "sources": sources,
                            },
                        )
                    )

                    # Execute the follow-up search
                    results = self._execute_search(next_query)
                    for result in results:
                        source = str(result.get("url", ""))
                        if source and source not in sources:
                            sources.append(source)

                        self.research_context.append(
                            {
                                "type": "search",
                                "query": next_query,
                                "content": str(result.get("text", "")),
                                "source": source,
                            }
                        )

                iteration += 1

                yield json.dumps(
                    self._create_step_result(
                        status="in_progress",
                        data="",
                        start_time=start_time,
                        metadata={
                            "logs": get_logs(),
                            "stage": "iteration_complete",
                            "iteration": f"{iteration}/{self.max_iterations}",
                            "sources": sources,
                        },
                    )
                )

            if iteration >= self.max_iterations:
                logger.info("⚠️  Reached research depth limit")

            # Synthesize findings
            logger.info("📚 Synthesizing research findings...")
            yield json.dumps(
                self._create_step_result(
                    status="in_progress",
                    data="",
                    start_time=start_time,
                    metadata={
                        "logs": get_logs(),
                        "stage": "synthesizing",
                        "sources": sources,
                    },
                )
            )

            final_context = "\n".join(
                [
                    f"{item['type']} - {item['source']} - {item['content']}"
                    for item in self.research_context
                ]
            )
            synthesis = self._synthesize_findings(resolved_prompt, final_context)

            logger.info("✅ Research complete!")

            # Final output with synthesis
            yield json.dumps(
                self._create_step_result(
                    status="success",
                    data=synthesis,
                    start_time=start_time,
                    metadata={
                        "logs": get_logs(),
                        "evaluation_statements": (
                            evaluation_statements["evaluation"]
                            if isinstance(evaluation_statements, dict)
                            and "evaluation" in evaluation_statements
                            else []
                        ),
                        "sources": sources,
                        "iterations": iteration + 1,
                        "total_sources": len(sources),
                        "criteria_met": isinstance(evaluation_statements, dict)
                        and "evaluation" in evaluation_statements
                        and all(
                            isinstance(stmt, dict) and stmt.get("status", "") == "pass"
                            for stmt in evaluation_statements["evaluation"]
                        ),
                        "stage": "complete",
                    },
                )
            )

        except Exception as e:
            logger.error(f"Error during research: {str(e)}")
            yield json.dumps(
                self._create_step_result(
                    status="error",
                    data=f"Research process failed: {str(e)}",
                    start_time=start_time,
                    metadata={
                        "logs": get_logs(),
                    },
                )
            )
