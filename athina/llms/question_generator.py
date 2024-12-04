from typing import List, Optional
from athina.llms.abstract_llm_service import AbstractLlmService
from athina.llms.openai_service import OpenAiService
from athina.interfaces.model import Model


class QuestionGenerator:
    _model: str
    _llm_service: AbstractLlmService

    """
    Generates closed-ended (Yes/No) questions given a  text.
    
    Attributes:
        n_questions (int): Number of questions to generate.
        openAIcompletion (OpenAICompletion): Instance for interactions with OpenAI's API.
    """

    # Pre-defined prompts for OpenAI's GPT model
    SYSTEM_MESSAGE = """ 
        You are an expert at generating closed-ended (Yes/No) questions given the content of a text.
    """

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step.
        1. Consider the text: {}.
        2. Generate {} closed-ended (Yes/No) questions based on the content.
        3. Return a JSON object in the following format: "question 1": 'Your question', "question 2": 'Your next question', ...
    """

    def __init__(
        self,
        model: str,
        n_questions: int,
        llm_service: Optional[AbstractLlmService] = None,
    ):
        """
        Initialize the QuestionGenerator.
        """
        self._model = model
        self.n_questions = n_questions

        if llm_service is None:
            self._llm_service = OpenAiService()
        else:
            self._llm_service = llm_service

    def generate(self, text: str) -> List[str]:
        """
        Generate a set of closed-ended questions based on the provided text.

        Args:
            text (str): The reference content used to generate questions.

        Returns:
            list[str]: A list of generated questions
        """
        user_message = self.USER_MESSAGE_TEMPLATE.format(text, self.n_questions)
        messages = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
        ]

        # Extract JSON object from LLM response
        json_response = self._llm_service.json_completion(
            model=self._model,
            messages=messages,
        )

        if json_response is None:
            raise Exception("Unable to generate questions")

        # Extract questions from JSON object
        questions = [question for question in json_response.values()]

        return questions
