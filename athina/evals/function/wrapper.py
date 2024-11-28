from typing import Optional, List
from athina.evals.eval_type import FunctionEvalTypeId
from athina.evals.function.function_evaluator import FunctionEvaluator


class ContainsAny(FunctionEvaluator):
    def __init__(
        self,
        keywords: List[str],
        case_sensitive: Optional[bool] = False,
        display_name: Optional[str] = None,
    ):
        """
        Initialize the ContainsAny function evaluator.

        Args:
            keywords (List[str]): List of keywords to check for in the text.
            case_sensitive (Optional[bool], optional): Whether the keyword matching should be case sensitive. Defaults to False.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.CONTAINS_ANY.value,
            function_arguments={"keywords": keywords, "case_sensitive": case_sensitive},
            display_name=display_name,
        )


class Regex(FunctionEvaluator):
    def __init__(
        self,
        pattern: str,
        display_name: Optional[str] = None,
    ):
        """
        Initialize the Regex function evaluator.

        Args:
            pattern (str): The regular expression pattern to be matched in the text.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.REGEX.value,
            function_arguments={"pattern": pattern},
            display_name=display_name,
        )


class ContainsNone(FunctionEvaluator):
    def __init__(
        self,
        keywords: List[str],
        case_sensitive: bool = False,
        display_name: Optional[str] = None,
    ):
        """
        Initialize the ContainsNone function evaluator.

        Args:
            keywords (str or List[str]): The keyword(s) to search for in the text.
            case_sensitive (bool, optional): If True, the comparison is case-sensitive. Defaults to False.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.CONTAINS_NONE.value,
            function_arguments={
                "keywords": keywords,
                "case_sensitive": case_sensitive,
            },
            display_name=display_name,
        )


class Contains(FunctionEvaluator):
    def __init__(
        self,
        keyword: str,
        case_sensitive: bool = False,
        display_name: Optional[str] = None,
    ):
        """
        Initialize the Contains function evaluator.

        Args:
            keyword (str): The keyword to search for in the text.
            case_sensitive (bool, optional): If True, the comparison is case-sensitive. Defaults to False.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.CONTAINS.value,
            function_arguments={
                "keyword": keyword,
                "case_sensitive": case_sensitive,
            },
            display_name=display_name,
        )


class ContainsAll(FunctionEvaluator):
    def __init__(
        self,
        keywords: List[str],
        case_sensitive: bool = False,
        display_name: Optional[str] = None,
    ):
        """
        Initialize the ContainsAll function evaluator.

        Args:
            keywords (List[str]): The list of keywords to search for in the text.
            case_sensitive (bool, optional): If True, the comparison is case-sensitive. Defaults to False.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.CONTAINS_ALL.value,
            function_arguments={
                "keywords": keywords,
                "case_sensitive": case_sensitive,
            },
            display_name=display_name,
        )


class ContainsJson(FunctionEvaluator):
    def __init__(self, display_name: Optional[str] = None):
        """
        Initialize the ContainsJson function evaluator.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.CONTAINS_JSON.value,
            function_arguments={},
        )


class ContainsEmail(FunctionEvaluator):
    def __init__(self, display_name: Optional[str] = None):
        """
        Initialize the ContainsEmail function evaluator.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.CONTAINS_EMAIL.value,
            function_arguments={},
            display_name=display_name,
        )


class IsJson(FunctionEvaluator):
    def __init__(self, display_name: Optional[str] = None):
        """
        Initialize the IsJson function evaluator.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.IS_JSON.value,
            function_arguments={},
            display_name=display_name,
        )


class IsEmail(FunctionEvaluator):
    def __init__(self, display_name: Optional[str] = None):
        """
        Initialize the IsEmail function evaluator.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.IS_EMAIL.value,
            function_arguments={},
            display_name=display_name,
        )


class NoInvalidLinks(FunctionEvaluator):
    def __init__(self, display_name: Optional[str] = None):
        """
        Initialize the NoInvalidLinks function evaluator.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.NO_INVALID_LINKS.value,
            function_arguments={},
            display_name=display_name,
        )


class ContainsLink(FunctionEvaluator):
    def __init__(self, display_name: Optional[str] = None):
        """
        Initialize the ContainsLink function evaluator.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.CONTAINS_LINK.value,
            function_arguments={},
            display_name=display_name,
        )


class ContainsValidLink(FunctionEvaluator):
    def __init__(self, display_name: Optional[str] = None):
        """
        Initialize the ContainsValidLink function evaluator.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.CONTAINS_VALID_LINK.value,
            function_arguments={},
            display_name=display_name,
        )


class Equals(FunctionEvaluator):
    def __init__(
        self,
        case_sensitive: bool = False,
        display_name: Optional[str] = None,
    ):
        """
        Initialize the Equals function evaluator.

        Args:
            case_sensitive (bool, optional): If True, the comparison is case-sensitive. Defaults to False.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.EQUALS.value,
            function_arguments={
                "case_sensitive": case_sensitive,
            },
            display_name=display_name,
        )


class StartsWith(FunctionEvaluator):
    def __init__(
        self,
        substring: str,
        case_sensitive: bool = False,
        display_name: Optional[str] = None,
    ):
        """
        Initialize the StartsWith function evaluator.

        Args:
            substring (str): The substring to check for at the start of the text.
            case_sensitive (bool, optional): If True, the comparison is case-sensitive. Defaults to False.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.STARTS_WITH.value,
            function_arguments={
                "substring": substring,
                "case_sensitive": case_sensitive,
            },
            display_name=display_name,
        )


class EndsWith(FunctionEvaluator):
    def __init__(
        self,
        substring: str,
        case_sensitive: bool = False,
        display_name: Optional[str] = None,
    ):
        """
        Initialize the EndsWith function evaluator.

        Args:
            substring (str): The substring to check for at the end of the text.
            case_sensitive (bool, optional): If True, the comparison is case-sensitive. Defaults to False.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.ENDS_WITH.value,
            function_arguments={
                "substring": substring,
                "case_sensitive": case_sensitive,
            },
            display_name=display_name,
        )


class LengthLessThan(FunctionEvaluator):
    def __init__(self, max_length: int, display_name: Optional[str] = None):
        """
        Initialize the LengthLessThan function evaluator.

        Args:
            max_length (int): The maximum length that the text should have.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.LENGTH_LESS_THAN.value,
            function_arguments={
                "max_length": max_length,
            },
            display_name=display_name,
        )


class LengthGreaterThan(FunctionEvaluator):
    def __init__(self, min_length: int, display_name: Optional[str] = None):
        """
        Initialize the LengthGreaterThan function evaluator.

        Args:
            min_length (int): The minimum length that the text should have.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.LENGTH_GREATER_THAN.value,
            function_arguments={
                "min_length": min_length,
            },
            display_name=display_name,
        )


class ApiCall(FunctionEvaluator):
    def __init__(
        self,
        url: str,
        payload: Optional[dict] = None,
        headers: Optional[dict] = None,
        display_name: Optional[str] = None,
    ):
        """
        Initialize the ApiCall function evaluator.

        Args:
            url (str): The URL to make the API call to.
            payload (dict): The payload to be sent in the API call. response, query, context, expected_response will be added to the payload.
            headers (dict, optional): The headers to be included in the API call. Defaults to None.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.API_CALL.value,
            function_arguments={
                "url": url,
                "payload": payload,
                "headers": headers,
            },
            display_name=display_name,
        )

class LengthBetween(FunctionEvaluator):
    def __init__(self, min_length: int, max_length: int, display_name: Optional[str] = None):
        """
        Initialize the LengthBetween function evaluator.

        Args:
            min_length (int): The minimum length that the text should have.
            max_length (int): The maximum length that the text should have.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.LENGTH_BETWEEN.value,
            function_arguments={
                "min_length": min_length,
                "max_length": max_length,
            },
            display_name=display_name,
        )

class OneLine(FunctionEvaluator):
    def __init__(self, display_name: Optional[str] = None):
        """
        Initialize the OneLine function evaluator.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.ONE_LINE.value,
            function_arguments={},
            display_name=display_name,
        )

class CustomCodeEval(FunctionEvaluator):
    def __init__(self, code: str, display_name: Optional[str] = None):
        """
        Initialize the Custom code evaluator.

        Args:
            code (str): The custom code to be executed.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.CUSTOM_CODE_EVAL.value,
            function_arguments={
                "code": code,
            },
            display_name=display_name,
        )

class JsonSchema(FunctionEvaluator):
    def __init__(self, schema: str, display_name: Optional[str] = None):
        """
        Initialize the JsonSchema function evaluator.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.JSON_SCHEMA.value,
            function_arguments={
                "schema": schema
            },
            display_name=display_name,
        )

class JsonValidation(FunctionEvaluator):
    def __init__(self, validations = None, display_name: Optional[str] = None):
        """
        Initialize the JsonValidation function evaluator.
        """
        super().__init__(
            function_name=FunctionEvalTypeId.JSON_VALIDATION.value,
            function_arguments={
                "validations": validations
            },
            display_name=display_name,
        )
