import argparse


class KeyValueAction(argparse.Action):
    """A custom action to parse key=value pairs into a dictionary."""

    def __call__(self, parser, namespace, values, option_string=None):
        kv_dict = {}
        for item in values:
            key, value = item.split("=", 1)  # Split only on the first '='
            kv_dict[key] = value
        setattr(namespace, self.dest, kv_dict)
