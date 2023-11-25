from ..example import FewShotExample, FewShotExampleInputParam

CONTEXT_CONTAINS_ENOUGH_INFORMATION_EXAMPLES = [
    FewShotExample(
        input_params=[
            FewShotExampleInputParam(
                name="context",
                value="Bjarne Stroustrup invented C++",
            ),
            FewShotExampleInputParam(
                name="query",
                value="Who invented the linux os?",
            ),
        ],
        eval_result="Fail",
        eval_reason="The context does not provide any relevant information about the Linux OS or its inventor.",
    ),
    FewShotExample(
        input_params=[
            FewShotExampleInputParam(
                name="context",
                value="In 1969, Neil Armstrong became the first person to walk on the moon.",
            ),
            FewShotExampleInputParam(
                name="query",
                value="What was the name of the spaceship used for the moon landing in 1969?",
            ),
        ],
        eval_result="Fail",
        eval_reason="The context provided does not include any information about the name of the spaceship used for the moon landing. The query specifically asks for the name of the spaceship, which is not present in the context.",
    ),
    FewShotExample(
        input_params=[
            FewShotExampleInputParam(
                name="context",
                value="YC is a seed stage accelerator program. It was founded in 2005 by Paul Graham, Jessica Livingston, Trevor Blackwell, and Robert Tappan Morris.",
            ),
            FewShotExampleInputParam(
                name="query",
                value="How much does YC invest in startups?",
            ),
        ],
        eval_result="Fail",
        eval_reason="The context does not include any information about the amount YC invests in startups.",
    ),
]
