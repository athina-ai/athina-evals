from evaluators.llm.example import FewShotExample, FewShotExampleInputParam

CONTEXT_RELEVANCE_EVAL_EXAMPLES = [
    FewShotExample(
        input_params=[
            FewShotExampleInputParam(
                name="user_query", value="Who is the president of USA?"
            ),
            FewShotExampleInputParam(
                name="context", value="The USA is a country in North America."
            ),
        ],
        eval_result="Fail",
        eval_reason="The response does not contain enough information to answer the user's query. The response does not mention the name of the president.",
    ),
    FewShotExample(
        input_params=[
            FewShotExampleInputParam(
                name="user_query",
                value="Who was the first person to set foot on the moon?",
            ),
            FewShotExampleInputParam(
                name="context",
                value="Apollo was the first spaceship to land on the moon",
            ),
        ],
        eval_result="Fail",
        eval_reason="The response does not contain enough information to answer the user's query. The response only talks about the spaceship that landed on the moon, but not the person.",
    ),
]
