#!/usr/bin/env python3

import argparse
from athina.helpers.config import ConfigHelper


def main():
    parser = argparse.ArgumentParser(
        prog="athina",
        description="Evaluation framework for your LLM-powered applications",
    )

    subparsers = parser.add_subparsers(title="commands", dest="command")

    # athina init
    parser_config = subparsers.add_parser("init", help="Configure settings")
    parser_config.set_defaults(func=init)

    # athina config
    parser_config = subparsers.add_parser("config", help="Configure settings")
    parser_config.set_defaults(func=config)

    # Other commands...

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


def init(args):
    """Initializes Athina and sets the necessary configuration variables"""
    config_data = ConfigHelper.load_config()

    openai_api_key = input("Enter your OpenAI API key: ")
    config_data["openai_api_key"] = openai_api_key

    athina_api_key = input("Enter your Athina API key: ")
    config_data["athina_api_key"] = athina_api_key

    # Add other configuration prompts as needed

    ConfigHelper.save_config(config_data)
    print("Configuration updated successfully. See athina_config.yml for details.")


def config(args):
    """Prints the current configuration"""
    config_data = ConfigHelper.load_config()
    print(config_data)


def run(args):
    pass


if __name__ == "__main__":
    main()
