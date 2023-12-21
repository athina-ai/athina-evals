from athina.interfaces.eval_type import RagasEvalTypeId
from athina.interfaces.metric import EvalMetric


class ContextContainsRelevantInformation():
    """
    This evaluator checks if the context contains relevant information to answer the user query.
    """

    SYSTEM_MESSAGE_TEMPLATE = """
    You are an expert at evaluating whether a chatbot can answer a user's query using ONLY the information provided to you as context.
    You are not concerned with factual correctness or accuracy. You only care whether the context contains enough information to answer the user's query.
    """

    USER_MESSAGE_TEMPLATE = """
        Let's think step by step:

        1. Consider the following: 
        user's query: {query}.
        context: {context}.
        2. Determine if the chatbot can answer the user's query with nothing but the "context" information provided to you.
        3. Provide a brief explanation of why the context does or does not contain sufficient information, labeled as 'explanation', leading up to a verdict (Pass/Fail) labeled as 'result'.
        4. Return a JSON object in the following format: "result": 'result', "explanation": 'explanation'.

        Here's are some examples: 
        {examples}
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return RagasEvalTypeId.CONTEXT_RELEVANCY.value

    @property
    def display_name(self):
        return "Context Contains Relevant Information"

    @property
    def metric_id(self) -> str:
        return EvalMetric.RELEVANCY_SCORE.value

    @property
    def default_model(self):
        return "gpt-4-1106-preview"

    @property
    def required_args(self):
        return ["query", "context"]

    @property
    def examples(self):
        return None

    def _user_message(
        self,
        query: str,
        context: str,
        **kwargs,
    ) -> str:
        return self.USER_MESSAGE_TEMPLATE.format(
            query=query,
            context=context,
            examples=self.examples,
        )