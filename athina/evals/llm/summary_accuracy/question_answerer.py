from athina.llms.openai_service import OpenAiService
from athina.interfaces.model import Model

class QuestionAnswerer:

    """
    This class determines whether the chatbot's answer was correct based on
    the given content and user's question.

    Attributes:
        openAIcompletion (OpenAICompletion): Instance for interactions with OpenAI's API.
    """

    # Pre-defined prompts for OpenAI's GPT model
    SYSTEM_MESSAGE = """ 
        You are an expert at responding to closed-ended (Yes/No) questions using ONLY the provided context.
    """

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step.
        1. Consider the following: 
           Questions: {}.
           Context: {}.
        2. Respond to each question from the provided 'questions', using either 
           'Yes', 'No', or 'Unknown', based ONLY on the given context. Do not use any information other than the context.
        3. Return a JSON object in the following format: "question1": "answer1", "question2": "answer2",...
    """

    def __init__(self, model):
        """
        Initialize the QuestionAnswerer class.
        """
        self._model = model
        self.openai_service = OpenAiService()

    def answer(self, questions: str, context: str) -> dict:
        """
        Respond to each question from the provided 'questions' given the context.

        Args:
            questions (str): A set of questions posed to the chatbot.
            context (str): Context used to inform the chatbot's answers.

        Returns:
            dict: Evaluation results formatted as a dictionary with questions as keys and
                  'Yes', 'No', or 'Unknown' as values.
        """

        user_message = self.USER_MESSAGE_TEMPLATE.format(questions, context)
        messages = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
        ]

        # Extract JSON object from LLM response
        json_response = self.openai_service.json_completion(
            model=self._model,
            messages=messages,
        )

        return json_response
