SYSTEM_MESSAGE = """You are given a list of messages from a conversation, with each message in the order it was sent. 

Your task is to analyze the flow of messages by the AI. For every message by the AI, follow these steps:

1. Read the message and consider it in the context of the previous messages in the conversation.

2. Think about the following:
- Does this message logically follow from the previous ones?
- Is there any contradiction or sudden shift in topic that makes this message seem out of place?

3. Decide if the message is logically "coherent" (it logically follows the conversation so far) or "not_coherent" (it breaks the logical flow or contradicts previous messages).

After considering each AI message through these steps, record your evaluation in a JSON object like this:

{ 
    "details": [ 
        {
            "message": message1,
            "result": "coherent / not_coherent",
            "explanation": “explanation of why this message is or is not coherent w.r.t previous messages"
        },
        ...
    ]
}

You must evaluate every single message in the conversation.
"""

USER_MESSAGE = """
Here is the conversation you need to evaluate:
{messages}
"""
