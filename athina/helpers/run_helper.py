import inspect
from athina import evals
from athina.interfaces.model import Model
from athina.helpers.config import ConfigHelper
from athina.keys import OpenAiApiKey, AthinaApiKey


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
    def validate_eval_args(eval_name, model, kwargs):
        """Validates the arguments for an eval"""

        # Check if eval_name is a valid eval
        available_evals = RunHelper.all_evals()
        if eval_name not in available_evals:
            raise ValueError(
                f"{eval_name} is not a valid eval.\n\nUse `athina list` to see all available evals."
            )

        # Check if model is in supported models
        if not Model.is_supported(model):
            raise ValueError(
                f"{model} is not a valid model.\n\nUse `athina models` to see all available models."
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
    def _set_keys():
        openai_api_key = ConfigHelper.load_openai_api_key()
        OpenAiApiKey.set_key(openai_api_key)

        athina_api_key = ConfigHelper.load_athina_api_key()
        AthinaApiKey.set_key(athina_api_key)

    @staticmethod
    def run_eval(eval_name, model, kwargs):
        """Runs an eval"""

        # Set the keys globally
        RunHelper._set_keys()

        # Validate the arguments for the eval
        if not RunHelper.validate_eval_args(eval_name, model, kwargs):
            # Handle invalid arguments, either by raising an exception or returning an error
            raise ValueError("Invalid arguments for the evaluation.")

        # Retrieve the evaluation class based on eval_name
        eval_class = getattr(evals, eval_name, None)

        # Check if the eval class exists and is a class
        if eval_class is None or not inspect.isclass(eval_class):
            raise ValueError(f"Invalid evaluation name: {eval_name}")

        # Instantiate the eval class with kwargs
        eval_instance = eval_class(model=model)

        # Run the evaluation - assuming there is a method like 'run' or similar
        result = eval_instance.run(**kwargs)

        # Return or handle the result as needed
        print(f"Completed running eval {eval_name} on {model}\n")
        print(result)
        return result
