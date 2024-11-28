# Step to classify text into one of the provided labels.
from typing import Union, Dict, Any
from athina.steps import Step
import marvin


class ClassifyText(Step):
    """
    Step that classifies text into one of the labels provided to the step.

    Attributes:
        input_column: The row's column to classify.
        labels: The labels to classify the text into.
        llm_api_key: The API key for the language model.
        language_model_id: The language model ID to use for classification.
    """

    labels: list[str]
    input_column: str
    llm_api_key: str
    language_model_id: str

    def execute(self, input_data: Any) -> Union[Dict[str, Any], None]:
        """Classify the text and return the label."""

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
            result = marvin.classify(
                input_text,
                labels=self.labels,
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
