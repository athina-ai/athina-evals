from typing import Optional
from athina.loaders import ResponseLoader, Loader


class LoaderHelper:
    """Helper class for loading data"""

    @staticmethod
    def get_loader(eval_name, loader_name: Optional[str] = None):
        """Returns the loader for the given format"""
        if (
            eval_name == "ContextContainsEnoughInformation"
            or eval_name == "DoesResponseAnswerQuery"
            or eval_name == "Faithfulness"
        ):
            return Loader
        else:
            if loader_name is None:
                raise ValueError(
                    f"Loader name must be specified for {eval_name} evaluation."
                )
            else:
                return ResponseLoader

    @staticmethod
    def load(eval_name, format, **kwargs):
        """Loads data based on the format specified."""
        loader = LoaderHelper.get_loader(eval_name)
        return loader().load(format, **kwargs)
