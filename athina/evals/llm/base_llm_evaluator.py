from typing import List, Optional
from athina.interfaces.result import LlmEvalResult
from athina.interfaces.model import Model
from athina.llms.openai_service import OpenAiService
from athina.helpers.logger import logger
from athina.helpers.json import JsonHelper
from athina.keys.openai_api_key import OpenAiApiKey
from .example import FewShotExample


class BaseLlmEvaluator:
    llm_service: OpenAiService
    grading_criteria: str

    REQUIRED_ARGS: List[str] = ["response"]

    TEMPERATURE = 0.0

    RETURN_FORMAT_INSTRUCTIONS = """
    Provide a brief explanation of why the response does or does not pass the grading criteria, labeled as 'explanation', leading up to a verdict (Pass/Fail) labeled as 'result'.

    You MUST return a JSON object in the following format: { "result": 'result', "explanation": 'explanation' }.
    """

    SYSTEM_MESSAGE_TEMPLATE = f""" 
    ### INSTRUCTIONS ###
    You are an expert at evaluating chatbot responses, according to some grading criteria.

    If it passes the grading criteria, then your result is Pass, otherwise it is Fail.

    {RETURN_FORMAT_INSTRUCTIONS}
    """

    USER_MESSAGE_TEMPLATE = """
    ### Grading Criteria ###
    {grading_criteria}

    ### EXAMPLES ###
    {examples}

    ### Response to evaluate ###
    {response}
    """

    EXAMPLES: FewShotExample = []

    def __init__(
        self,
        model: str,
        metadata: Optional[dict] = None,
        grading_criteria: Optional[str] = None,
    ):
        self.llm_service = OpenAiService()
        self.grading_criteria = grading_criteria if grading_criteria else ""
        if not Model.is_supported(model):
            raise ValueError(f"Unsupported model: {model}")

    def _examples_str(self) -> str:
        return "\n".join([str(example) for example in self.EXAMPLES])

    def _system_message(self, **kwargs) -> str:
        return self.SYSTEM_MESSAGE_TEMPLATE

    def _user_message(self, **kwargs) -> str:
        return self.USER_MESSAGE_TEMPLATE.format(
            examples=self._examples_str(),
            grading_criteria=self.grading_criteria,
            **kwargs,
        )

    def _prompt_messages(self, **kwargs) -> List[dict]:
        return [
            {
                "role": "system",
                "content": self._system_message(**kwargs),
            },
            {
                "role": "user",
                "content": self._user_message(**kwargs),
            },
        ]

    def _validate_args(self, **kwargs) -> None:
        for arg in self.REQUIRED_ARGS:
            if arg not in kwargs:
                raise ValueError(f"Missing required argument: {arg}")

    def run(self, **kwargs) -> LlmEvalResult:
        self._validate_args(**kwargs)
        messages = self._prompt_messages(**kwargs)

        chat_completion_response = self.llm_service.chat_completion(
            messages=messages,
            temperature=self.TEMPERATURE,
        )
        chat_completion_response_json = JsonHelper.extract_json_from_text(
            chat_completion_response
        )

        try:
            result = chat_completion_response_json["result"]
            explanation = chat_completion_response_json["explanation"]
            failure = result == "Fail"
        except KeyError as e:
            logger.error(f"Key missing in the response JSON: {e}")
            raise ValueError(f"Response JSON is missing a required key: {e}")

        return LlmEvalResult(
            failure=failure,
            reason=explanation,
        )
