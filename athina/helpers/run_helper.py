import inspect
from athina import evals


class RunHelper:
    @staticmethod
    def all_evals():
        # List to store the names of classes
        exported_classes = []

        # Iterate through each attribute in the module
        for name in dir(evals):
            # Get the attribute
            attribute = getattr(evals, name)

            # Check if the attribute is a class and is listed in __all__
            if inspect.isclass(attribute) and name in evals.__all__:
                exported_classes.append(name)

        # Return the names of the exported classes
        return exported_classes

    @staticmethod
    def validate_eval_args(eval_name, kwargs):
        """Validates the arguments for an eval"""

        available_evals = RunHelper.all_evals()
        if eval_name not in available_evals:
            raise ValueError(
                f"{eval_name} is not a valid eval.\n\nUse `athina list` to see all available evals."
            )

        # Retrieve the evaluation class based on eval_name
        eval_class = getattr(evals, eval_name, None)

        # Check if the eval class exists
        if eval_class is None or not inspect.isclass(eval_class):
            raise ValueError(f"Invalid evaluation name: {eval_name}")

        # Retrieve the required arguments from the eval class
        required_args = getattr(eval_class, "REQUIRED_ARGS", [])

        # Check if each required argument is in kwargs
        missing_args = [arg for arg in required_args if arg not in kwargs]
        if missing_args:
            raise ValueError(
                f"Missing required arguments for {eval_name}: {', '.join(missing_args)}"
            )

        # If all required arguments are present, return True or some confirmation
        return True

    @staticmethod
    def run_eval(eval_name, kwargs):
        """Runs an eval"""

        pass
