from ..example import FewShotExample, FewShotExampleInputParam

FAITHFULNESS_EVAL_EXAMPLES = [
    FewShotExample(
        input_params=[
            FewShotExampleInputParam(
                name="context",
                value="Y Combinator is a startup accelerator launched in March 2005. It has been used to launch more than 4,000 companies",
            ),
            FewShotExampleInputParam(name="response", value="$125,000"),
        ],
        eval_result="Fail",
        eval_reason="The response cannot be inferred from the provided context. The context does not mention the $125,000 figure.",
    ),
]
