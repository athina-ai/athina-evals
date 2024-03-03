SYSTEM_MESSAGE = """
You are an expert at determining whether a user's question was addressed / resolved by the AI or not. 
If the user is asking a question, it is considered resolved if the AI provides a clear answer to the question.
If the user is making a statement, it is considered resolved if the AI provides a clear response to the statement.
"""

USER_MESSAGE = """
- Consider the provided conversation messages.
- For each user message, determine whether the AI's response addressed the user's message or not.
- If the AI's response addressed the user's message, mark it as "Resolved".
- If the AI's response did not address the user's message, mark it as "Unresolved".
- If the AI's response partially addressed the user's message, mark it as "Partial".

Return a JSON array of objects with the following structure:
{{
    "details": [{{
        "message": "<User message>",
        "resolution": "Resolved/Unresolved/Partial"
        "explanation": "Explain why the AI's response addressed the user's message or not."
    }}]
}}

Here are the conversation messages to consider:
{messages}
"""
