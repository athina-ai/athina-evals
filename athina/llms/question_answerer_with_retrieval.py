import numpy as np
import pprint
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
from athina.llms.abstract_llm_service import AbstractLlmService
from athina.llms.openai_service import OpenAiService
from concurrent.futures import ThreadPoolExecutor, as_completed
from .question_answerer import QuestionAnswerer, QuestionAnswererResponse

class ContextFinderStrategy(ABC):

    @abstractmethod
    def find_relevant_context_index(self, question, context_chunks):
        pass


class EmbeddingBasedContextFinder(ContextFinderStrategy):

    def __init__(self, preprocessed_context_embeddings):
        self.preprocessed_context_embeddings = preprocessed_context_embeddings

    @staticmethod
    def cosine_similarity(vec_a, vec_b):
        # Convert to numpy arrays and check if they are numeric
        vec_a = np.asarray(vec_a, dtype=np.float32)
        vec_b = np.asarray(vec_b, dtype=np.float32)

        if np.all(vec_a == 0) or np.all(vec_b == 0):
            return 0

        dot_product = np.dot(vec_a, vec_b)
        magnitude_a = np.linalg.norm(vec_a)
        magnitude_b = np.linalg.norm(vec_b)

        return dot_product / (magnitude_a * magnitude_b)
    
    def find_relevant_context_indices(self, question_embedding, context_embeddings, num_relevant=5):
        # Ensure context_embeddings is a list of numpy arrays
        context_embeddings = [np.asarray(embedding) for embedding in context_embeddings]

        # Compute cosine similarities
        similarities = [EmbeddingBasedContextFinder.cosine_similarity(question_embedding, context_embedding) for context_embedding in context_embeddings]

        # Find the indices of the top 'num_relevant' most similar context chunks
        relevant_indices = np.argsort(similarities)[-num_relevant:][::-1]
        return relevant_indices

    def find_relevant_context_index(self, question_embedding, context_embeddings):
        self.find_relevant_context_indices(question_embedding, context_embeddings, num_relevant=1)[0]


class QuestionAnswererWithRetrieval(QuestionAnswerer):

    _llm_service: AbstractLlmService

    SYSTEM_MESSAGE = """ 
        You are an expert at responding to closed-ended (Yes/No) questions using ONLY the provided context.
        You MUST return the response as a JSON object with 3 fields: question, answer, and explanation
    """

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step.
        1. Consider the following: 
           Question: '{}'.
           Context: '{}'.
        2. Based on the context provided, think through the question and determine an explanation for your response.
        3. If you cannot determine an answer, respond with 'Unknown'.
        4. Respond to the question with an explanation, leading up to a final answer to the question: 'Yes', 'No', or 'Unknown'.
        5. Return a JSON object in the following format: "answer": "answer", "explanation": "explanation"
    """

    def __init__(self, 
        context, 
        model: str = "gpt-4-1106-preview", 
        llm_service: Optional[AbstractLlmService] = None,
        context_chunk_size=128
    ):
        self._model = model
        if llm_service is None:
            self._llm_service = OpenAiService()
        else:
            self._llm_service = llm_service
        self.context_chunks, self.context_embeddings = self._preprocess_context(context, context_chunk_size)
        self.context_finder = EmbeddingBasedContextFinder(self.context_embeddings)

    def _preprocess_context(self, context, chunk_size):
        # Split context into chunks of specified size
        # This is a placeholder; implement your chunking logic based on your requirements
        context_chunks = [context[i:i+chunk_size] for i in range(0, len(context), chunk_size)]
        
        # Generate embeddings for each context chunk
        context_embeddings = [self._llm_service.embeddings(chunk) for chunk in context_chunks]
        return context_chunks, context_embeddings

    def _get_relevant_chunks(self, question):
        ADJACENT_CHUNKS = 1
        question_embedding = self._llm_service.embeddings(question)
        relevant_context_indices = self.context_finder.find_relevant_context_indices(question_embedding, self.context_embeddings, num_relevant=3)
        relevant_context_chunks = []
        for idx in relevant_context_indices:
            min_idx = max(0, idx-ADJACENT_CHUNKS)
            max_idx = min(len(self.context_chunks), idx+ADJACENT_CHUNKS)
            relevant_context_chunks.append("".join(self.context_chunks[min_idx:max_idx]))
        
        return relevant_context_chunks

    def _answer_question(self, question) -> QuestionAnswererResponse:
        relevant_context_chunks = self._get_relevant_chunks(question)
        relevant_context = "\n".join(relevant_context_chunks)
        
        user_message = self.USER_MESSAGE_TEMPLATE.format(question, relevant_context)
        messages = [
            {"role": "system", "content": self.SYSTEM_MESSAGE},
            {"role": "user", "content": user_message},
        ]

        # Extract JSON object from LLM response for a single question
        json_completion = self._llm_service.json_completion(
            model=self._model,
            messages=messages,
        )

        if json_completion is None:
            raise Exception("No response from LLM")
        
        try:
            answer = json_completion["answer"]
            explanation = json_completion["explanation"]

            return {
                "answer": answer,
                "explanation": explanation,
            }
        except:
            return {
                "answer": "Error",
                "explanation": None,
            }


    def answer(self, questions: List[str], **kwargs) -> Tuple[dict, dict]:
        results = {}
        simple_result = {}
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._answer_question, question): question for question in questions
            }

            for future in as_completed(futures):
                question = futures[future]
                try:
                    response = future.result()
                    results[question] = response
                    simple_result[question] = response["answer"]
                except Exception as exc:
                    print(f'Question {question} generated an exception: {exc}')
                    results[question] = {
                        "answer": "Error",
                        "explanation": None,
                    }
                    simple_result[question] = 'Error'

        return results, simple_result