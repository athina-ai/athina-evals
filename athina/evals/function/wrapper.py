
from typing import Optional, List
from athina.evals.eval_type import FunctionEvalTypeId
from athina.evals.function.function_evaluator import FunctionEvaluator


class ContainsAny(FunctionEvaluator):
    def __init__(
        self,
        keywords: List[str],
        case_sensitive: Optional[bool] = False,
    ):
        """
        Initialize the ContainsAny function evaluator.

        Args:
            keywords (List[str]): List of keywords to check for in the response.
            case_sensitive (Optional[bool], optional): Whether the keyword matching should be case sensitive. Defaults to False.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.CONTAINS_ANY.value,
            function_arguments={"keywords":keywords, "case_sensitive":case_sensitive},
        )

class Regex(FunctionEvaluator):
    def __init__(
        self,
        regex: str,
    ):
        """
        Initialize the Regex function evaluator.

        Args:
            regex (str): The regular expression pattern to be matched in the response.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.REGEX.value,
            function_arguments={"pattern":regex},
        )

class ContainsNone(FunctionEvaluator):
    def __init__(self, keywords: List[str], case_sensitive: bool = False):
        """
        Initialize the ContainsNone function evaluator.

        Args:
            keywords (str or List[str]): The keyword(s) to search for in the response.
            case_sensitive (bool, optional): If True, the comparison is case-sensitive. Defaults to False.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.CONTAINS_NONE.value,
            function_arguments={
                "keywords": keywords,
                "case_sensitive": case_sensitive,
            },
        )

class Contains(FunctionEvaluator):
    def __init__(self, keyword: str, case_sensitive: bool = False):
        """
        Initialize the Contains function evaluator.

        Args:
            keyword (str): The keyword to search for in the response.
            case_sensitive (bool, optional): If True, the comparison is case-sensitive. Defaults to False.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.CONTAINS.value,
            function_arguments={
                "keyword": keyword,
                "case_sensitive": case_sensitive,
            },
        )

class ContainsAll(FunctionEvaluator):
    def __init__(self, keywords: List[str], case_sensitive: bool = False):
        """
        Initialize the ContainsAll function evaluator.

        Args:
            keywords (List[str]): The list of keywords to search for in the response.
            case_sensitive (bool, optional): If True, the comparison is case-sensitive. Defaults to False.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.CONTAINS_ALL.value,
            function_arguments={
                "keywords": keywords,
                "case_sensitive": case_sensitive,
            },
        )

class ContainsJson(FunctionEvaluator):
    def __init__(self):
        """
        Initialize the ContainsJson function evaluator.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.CONTAINS_JSON.value,
            function_arguments={},
        )

class ContainsEmail(FunctionEvaluator):
    def __init__(self):
        """
        Initialize the ContainsEmail function evaluator.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.CONTAINS_EMAIL.value,
            function_arguments={},
        )

class IsJson(FunctionEvaluator):
    def __init__(self):
        """
        Initialize the IsJson function evaluator.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.IS_JSON.value,
            function_arguments={},
        )

class IsEmail(FunctionEvaluator):
    def __init__(self):
        """
        Initialize the IsEmail function evaluator.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.IS_EMAIL.value,
            function_arguments={},
        )

class NoInvalidLinks(FunctionEvaluator):
    def __init__(self):
        """
        Initialize the NoInvalidLinks function evaluator.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.NO_INVALID_LINKS.value,
            function_arguments={},
        )

class ContainsLink(FunctionEvaluator):
    def __init__(self):
        """
        Initialize the ContainsLink function evaluator.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.CONTAINS_LINK.value,
            function_arguments={},
        )

class ContainsValidLink(FunctionEvaluator):
    def __init__(self):
        """
        Initialize the ContainsValidLink function evaluator.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.CONTAINS_VALID_LINK.value,
            function_arguments={},
        )

class Equals(FunctionEvaluator):
    def __init__(self, expected_response: str, case_sensitive: bool = False):
        """
        Initialize the Equals function evaluator.

        Args:
            expected_response (str): The expected response to compare against.
            case_sensitive (bool, optional): If True, the comparison is case-sensitive. Defaults to False.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.EQUALS.value,
            function_arguments={
                "expected_response": expected_response,
                "case_sensitive": case_sensitive,
            },
        )

class StartsWith(FunctionEvaluator):
    def __init__(self, substring: str, case_sensitive: bool = False):
        """
        Initialize the StartsWith function evaluator.

        Args:
            substring (str): The substring to check for at the start of the response.
            case_sensitive (bool, optional): If True, the comparison is case-sensitive. Defaults to False.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.STARTS_WITH.value,
            function_arguments={
                "substring": substring,
                "case_sensitive": case_sensitive,
            },
        )

class EndsWith(FunctionEvaluator):
    def __init__(self, substring: str, case_sensitive:bool = False):
        """
        Initialize the EndsWith function evaluator.

        Args:
            substring (str): The substring to check for at the end of the response.
            case_sensitive (bool, optional): If True, the comparison is case-sensitive. Defaults to False.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.ENDS_WITH.value,
            function_arguments={
                "substring": substring,
                "case_sensitive": case_sensitive,
            },
        )

class LengthLessThan(FunctionEvaluator):
    def __init__(self, max_length: int):
        """
        Initialize the LengthLessThan function evaluator.

        Args:
            max_length (int): The maximum length that the response should have.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.LENGTH_LESS_THAN.value,
            function_arguments={"max_length": max_length, },
        )

class LengthGreaterThan(FunctionEvaluator):
    def __init__(self, min_length: int):
        """
        Initialize the LengthGreaterThan function evaluator.

        Args:
            min_length (int): The minimum length that the response should have.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.LENGTH_GREATER_THAN.value,
            function_arguments={"min_length": min_length, },
        )

class ApiCall(FunctionEvaluator):
    def __init__(self, url: str, payload: dict = None, headers: dict = None):
        """
        Initialize the ApiCall function evaluator.

        Args:
            url (str): The URL to make the API call to.
            payload (dict): The payload to be sent in the API call. Response will be added to the dict as "response".
            headers (dict, optional): The headers to be included in the API call. Defaults to None.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.API_CALL.value,
            function_arguments={
                "url": url,
                "payload": payload,
                "headers": headers,
            },
        )