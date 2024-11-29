from typing import List, Tuple, Optional
from athina.llms.openai_service import OpenAiService
from athina.llms.abstract_llm_service import AbstractLlmService
from .question_answerer import QuestionAnswerer


class QuestionAnswererBulk(QuestionAnswerer):

    _llm_service: AbstractLlmService

    """
    This class responds to a list of closed-ended (Y/N) questions based on a provided context.
    It does so using a single LLM inference call, and retrieving a JSON dictionary of all responses.
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
           'Yes', 'No', or 'Unknown', based ONLY on the given context.
        3. Return a JSON object in the following format: 
            [question1]: answer1,
            [question2]: answer2,
            ...
    """

    def __init__(
        self,
        model: str = "gpt-4-1106-preview",
        llm_service: Optional[AbstractLlmService] = None,
    ):
        """
        Initialize the QuestionAnswerer class.
        """
        self._model = model
        if llm_service is None:
            self._llm_service = OpenAiService()
        else:
            self._llm_service = llm_service

    def answer(self, questions: List[str], context: str) -> Tuple[dict, dict]:
        """
        Respond to each question from the provided 'questions' given the context.
        """

        questions_str = "\n".join(questions)
        user_message = self.USER_MESSAGE_TEMPLATE.format(questions_str, context)
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
            raise Exception("No response from LLM")

        output = {}
        simple_output = {}
        for i in range(len(questions)):
            question = questions[i]
            try:
                answer = json_response[question]
                output[question] = {"answer": answer, "explanation": None}
                simple_output[question] = answer
            except:
                output[question] = {
                    "answer": "Error",
                    "explanation": None,
                }
                simple_output[question] = "Error"

        return output, simple_output
