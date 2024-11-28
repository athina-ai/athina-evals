# Step to extract entities from text using the instructions.
from typing import Union, Dict, Any
from athina.steps import Step
import marvin


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

        if input_data is None:
            input_data = {}

        if not isinstance(input_data, dict):
            raise TypeError("Input data must be a dictionary.")

        input_text = input_data.get(self.input_column, None)
        
        if input_text is None:
            return None
        
        marvin.settings.openai.api_key = self.llm_api_key
        marvin.settings.openai.chat.completions.model = self.language_model_id
        
        try:
            result = marvin.extract(
                input_text,
                instructions=self.instructions,
            )
            return {
                "status": "success",
                "data": result,
            }
        except Exception as e:
            return {
                "status": "error",
                "data": str(e),
            }
