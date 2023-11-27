import inspect
from athina import evals
from athina.interfaces.model import Model
from athina.helpers.config import ConfigHelper
from athina.helpers.loader_helper import LoaderHelper
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
    def get_evaluator(eval_name, **kwargs):
        """Returns an evaluator class based on the eval name"""

        # Retrieve the evaluation class based on eval_name
        eval_class = getattr(evals, eval_name, None)

        # Check if the eval class exists and is a class
        if eval_class is None or not inspect.isclass(eval_class):
            raise ValueError(f"Invalid evaluation name: {eval_name}")

        return eval_class(**kwargs)

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
        evaluator = RunHelper.get_evaluator(eval_name, model=model)

        # Check if the eval class exists
        if evaluator is None:
            raise ValueError(f"Invalid evaluation name: {eval_name}")

        # Retrieve the required arguments from the eval class
        required_args = evaluator.required_args()

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

        print(f"Running eval {eval_name} on {model}...\n")
        # Set the keys globally
        RunHelper._set_keys()

        # Validate the arguments for the eval
        if not RunHelper.validate_eval_args(eval_name, model, kwargs):
            # Handle invalid arguments, either by raising an exception or returning an error
            raise ValueError("Invalid arguments for the evaluation.")

        # Retrieve the evaluation class based on eval_name
        evaluator = RunHelper.get_evaluator(eval_name, model=model)

        # Run the evaluation
        result = evaluator.run(**kwargs)

        # Return or handle the result as needed
        runtime = float(result["runtime"] / 1000)
        print(f"Completed running eval {eval_name} on {model} in {runtime} seconds\n")
        print(result)
        return result

    @staticmethod
    def run_eval_on_batch(eval_name, model, format, **kwargs):
        """Runs an eval on a batch dataset"""

        print(f"Running eval {eval_name} on dataset using {model}...\n")
        # Set the keys globally
        RunHelper._set_keys()

        # TODO: Validate the arguments for the eval

        # Get loader
        loader = LoaderHelper.get_loader(eval_name)()
        dataset = loader.load(format, **kwargs)

        # Retrieve the evaluation class based on eval_name
        evaluator = RunHelper.get_evaluator(eval_name, model=model)

        # Run the batch evaluation
        result = evaluator.run_batch(data=dataset)

        # Return or handle the result as needed
        runtime = float(result["runtime"] / 1000)
        print(f"Completed running eval {eval_name} on {model} in {runtime} seconds\n")
        print(result)
        return result
