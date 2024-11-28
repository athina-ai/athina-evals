import inspect


def get_named_parameters(func):
    """
    Get all named parameters of a function.
    """
    parameters = inspect.signature(func).parameters
    named_parameters = [
        param
        for param in parameters
        if parameters[param].default != inspect.Parameter.empty
    ]
    return named_parameters


def get_named_non_default_parameters(func):
    """
    Get all named parameters without default values of a function.
    """
    parameters = inspect.signature(func).parameters
    named_non_default_parameters = [
        param
        for param in parameters
        if parameters[param].default == inspect.Parameter.empty
    ]
    return named_non_default_parameters
