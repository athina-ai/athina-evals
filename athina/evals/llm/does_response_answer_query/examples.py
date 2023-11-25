from ..example import FewShotExample, FewShotExampleInputParam

DOES_RESPONSE_ANSWER_QUERY_EVAL_EXAMPLES = [
    FewShotExample(
        input_params=[
            FewShotExampleInputParam(
                name="query", value="Who was the first person to land on the moon?"
            ),
            FewShotExampleInputParam(
                name="response",
                value="The Apollo 11 was the first spaceship to land on the moon.",
            ),
        ],
        eval_result="Fail",
        eval_reason="The response does not answer the user's query sufficiently. It mentions the Apollo 11 spaceship, but does not mention the name of the astronaut.",
    ),
    FewShotExample(
        input_params=[
            FewShotExampleInputParam(
                name="query", value="Who was the first person to land on the moon?"
            ),
            FewShotExampleInputParam(
                name="response",
                value="I'm sorry, I don't know the answer to that question.",
            ),
        ],
        eval_result="Fail",
        eval_reason="The response does not answer the user's query. It simply states that it does not know the answer.",
    ),
]
