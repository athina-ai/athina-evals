# Step to extract entities from text using the instructions.
from typing import Union, Dict, Any
from athina.steps import Step
import marvin
import time


class ExtractEntities(Step):
    """
    Step that extracts entities from text using the instructions provided to the step.

    Attributes:
        input_column: The row's column to extract entities from.
        instructions: The instructions to extract entities from the text.
        llm_api_key: The API key for the language model.
        language_model_id: The language model ID to use for entity extraction.
    """

    input_column: str
    instructions: str
    llm_api_key: str
    language_model_id: str

    def execute(self, input_data: Any) -> Union[Dict[str, Any], None]:
        """Extract entities from the text and return the entities."""
        start_time = time.perf_counter()
        if input_data is None:
            input_data = {}

        if not isinstance(input_data, dict):
            return self._create_step_result(
                status="error",
                data="Input data must be a dictionary.",
                start_time=start_time,
            )
        input_text = input_data.get(self.input_column, None)

        if input_text is None:
            return self._create_step_result(
                status="error",
                data="Input column not found.",
                start_time=start_time,
            )

        marvin.settings.openai.api_key = self.llm_api_key
        marvin.settings.openai.chat.completions.model = self.language_model_id

        try:
            result = marvin.extract(
                input_text,
                instructions=self.instructions,
            )
            return self._create_step_result(
                status="success",
                data=result,
                start_time=start_time,
            )
        except Exception as e:
            return self._create_step_result(
                status="error",
                data=str(e),
                start_time=start_time,
            )
