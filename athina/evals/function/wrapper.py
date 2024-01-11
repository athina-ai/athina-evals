
from typing import Optional
from athina.evals.eval_type import FunctionEvalTypeId
from athina.evals.function.function_evaluator import FunctionEvaluator


class ContainsAny(FunctionEvaluator):
    def __init__(
        self,
        keywords: dict,
        case_sensitive: Optional[bool] = False,
    ):
        super().__init__(
            function_name=FunctionEvalTypeId.CONTAINS_ANY.value,
            function_arguments={"keywords":keywords, "case_sensitive":case_sensitive},
        )

class Regex(FunctionEvaluator):
    def __init__(
        self,
        regex: str,
    ):
        super().__init__(
            function_name=FunctionEvalTypeId.REGEX.value,
            function_arguments={"pattern":regex},
        )