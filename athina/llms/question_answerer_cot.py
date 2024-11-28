from typing import Tuple, List, Optional
from athina.llms.abstract_llm_service import AbstractLlmService
from .question_answerer import QuestionAnswerer, QuestionAnswererResponse
from athina.llms.openai_service import OpenAiService

class QuestionAnswererChainOfThought(QuestionAnswerer):

    _llm_service: AbstractLlmService

    """
    This class responds to a list of closed-ended (Y/N) questions based on a provided context.
    It does so using a separate LLM inference call with CoT prompting for each question.
    It also asks the LLM to provide an explanation for each answer, which helps improve the reasoning.
    """

    # Pre-defined prompts for OpenAI's GPT model
    SYSTEM_MESSAGE = """ 
        You are an expert at responding to closed-ended (Yes/No) questions using the provided context.
    """

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step.
        1. Consider the following: 
           Question: {}.
           Context: {}.
        2. Based on the context provided, think through the question and determine an explanation for your response.
        3. If you cannot determine an answer, respond with 'Unknown'.
        4. Respond to the question with an explanation, leading up to a final answer to the question: 'Yes', 'No', or 'Unknown'.
        5. Return a JSON object in the following format: "answer": "answer", "explanation": "explanation"
    """

    def __init__(self, 
        model: str = "gpt-4-1106-preview", 
        llm_service: Optional[AbstractLlmService] = None
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

        results = {}
        simple_output = {}
        for question in questions:
            try:
                response = self.answer_question(question, context)
                results[question] = response
                simple_output[question] = response["answer"]
            except:
                results[question] = {
                    "answer": "Error",
                    "explanation": None,
                }
                simple_output[question] = "Error"
        return results, simple_output

    def answer_question(self, question: str, context: str) -> QuestionAnswererResponse:
        """
        Respond to each question from the provided 'questions' given the context.

        Args:
            question (str): A set of questions posed to the chatbot.
            context (str): Context used to inform the chatbot's answers.

        Returns:
            dict: Evaluation results formatted as a dictionary with questions as keys and
                  'Yes', 'No', or 'Unknown' as values.
        """

        user_message = self.USER_MESSAGE_TEMPLATE.format(question, context)
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

        answer = json_response["answer"]
        explanation = json_response["explanation"]

        return {
            "answer": answer,
            "explanation": explanation,
        }
