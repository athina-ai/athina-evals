# Step to chat with OpenAI's Assistant API.
from typing import Union, Dict, Any
from athina.steps import Step
from openai import OpenAI
import os


class OpenAiAssistant(Step):
    """
    Step that chats with OpenAI's Assistant API.

    Attributes:
        assistant_id: The assistant ID to be used.
        openai_api_key: OpenAI's API Key.
        input_column: The row's column to classify.
    """
    assistant_id: str
    openai_api_key: str
    input_column: str
    client: any = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, assistant_id: str, openai_api_key: str, input_column: str):
        super().__init__(assistant_id=assistant_id, openai_api_key=openai_api_key, input_column=input_column)
        self.client = OpenAI(api_key=openai_api_key)

    def execute(self, input_data: Any) -> Union[Dict[str, Any], None]:
        """Calls OpenAI's Assistant API and returns the response."""

        if input_data is None:
            input_data = {}

        if not isinstance(input_data, dict):
            raise TypeError("Input data must be a dictionary.")

        input_text = input_data.get(self.input_column, None)
        
        if input_text is None:
            return None
        try:
            # Create a thread
            thread = self.client.beta.threads.create()

            # Add input_text to the thread
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role='user',
                content=input_text
            )

            # Run the assistant
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )

            # Wait for the run to complete
            while run.status not in ["completed", "failed"]:
                run = self.client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

            # Handle failed case
            if run.status == "failed":
                return {
                    "status": "error",
                    "data": "The assistant run failed."
                }
            
            # Retrieve the assistant's response
            messages = self.client.beta.threads.messages.list(thread_id=thread.id)
            
            # Check and return the assistant's response based on format
            for message in messages.data:
                if message.role == "assistant":
                    for content in message.content:
                        if content.type == "text":
                            return {"status": "success", "data": content.text.value}
                        elif content.type == "json":
                            return {"status": "success", "data": content.json.value}
            return {
                "status": "success",
                "data": None,
            }
        except Exception as e:
            return {
                "status": "error",
                "data": str(e),
            }
