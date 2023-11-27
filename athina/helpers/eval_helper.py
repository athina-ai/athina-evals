from athina.evals import __all__ as supported_evals


class EvalHelper:
    @staticmethod
    def is_supported(eval_name: str):
        return eval_name in supported_evals
