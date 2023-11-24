from ..example import FewShotExample, FewShotExampleInputParam

FAITHFULNESS_EVAL_EXAMPLES = [
    FewShotExample(
        input_params=[
            FewShotExampleInputParam(
                name="context",
                value="Y Combinator is a startup accelerator launched in March 2005. It has been used to launch more than 4,000 companies.",
            ),
            FewShotExampleInputParam(
                name="response",
                value="YC invests $125,000 in startups in exchange for equity.",
            ),
        ],
        eval_result="Fail",
        eval_reason="The response cannot be inferred from the provided context. The context does not mention that YC invests $125,000 in startups.",
    ),
    FewShotExample(
        input_params=[
            FewShotExampleInputParam(
                name="context",
                value="The president of the United States is Joe Biden.",
            ),
            FewShotExampleInputParam(
                name="response",
                value="Barack Obama was the 44th president of the United States.",
            ),
        ],
        eval_result="Fail",
        eval_reason="The response cannot be inferred from the provided context. The context does not state anything that suggests Barack Obama was the 44th president of the United States.",
    ),
]
