
from typing import Optional
from athina.evals.function.function_evaluator import FunctionEvaluator


class ContainsAny(FunctionEvaluator):
    def __init__(
        self,
        keywords: dict,
        case_sensitive: Optional[bool] = False,
    ):
        super().__init__(
            function_name="ContainsAny",
            function_arguments={"keywords":keywords, "case_sensitive":case_sensitive},
        )

class Regex(FunctionEvaluator):
    def __init__(
        self,
        regex: str,
    ):
        super().__init__(
            function_name="Regex",
            function_arguments={"pattern":regex},
        )