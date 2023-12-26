
class TypeCheckHelper:
    @staticmethod
    def is_valid_type(value, expected_type) -> bool:
        """
        Check if the value is of the expected type.
        """
        if hasattr(expected_type, '__origin__'):  # For types like List[str]
            if not isinstance(value, expected_type.__origin__):
                return False
            element_type = expected_type.__args__[0]
            return all(isinstance(elem, element_type) for elem in value)
        else:  # For regular types like str
            return isinstance(value, expected_type)