from ..example import FewShotExample, FewShotExampleInputParam

ANSWER_RELEVANCE_EVAL_EXAMPLES = [
    FewShotExample(
        input_params=[
            FewShotExampleInputParam(
                name="user_query", value="Who is the president of USA?"
            ),
            FewShotExampleInputParam(
                name="response", value="I'm not sure who the president is"
            ),
        ],
        eval_result="Fail",
        eval_reason="The response does not answer the user's query sufficiently because it does not mention the name of the president.",
    )
]
