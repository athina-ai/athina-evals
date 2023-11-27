#!/usr/bin/env python3

import argparse
from athina.helpers.config import ConfigHelper
from athina.helpers.run_helper import RunHelper
from athina.helpers.kwparser import KeyValueAction


def main():
    parser = argparse.ArgumentParser(
        prog="athina",
        description="Evaluation framework for your LLM-powered applications",
    )

    subparsers = parser.add_subparsers(title="commands", dest="command")

    # athina init
    parser_init = subparsers.add_parser("init", help="Configure settings")
    parser_init.set_defaults(func=init)

    # athina config
    parser_config = subparsers.add_parser("config", help="Configure settings")
    parser_config.set_defaults(func=config)

    # athina list
    parser_config = subparsers.add_parser("list", help="Lists all available evals")
    parser_config.set_defaults(func=list)

    # athina run [eval_name] --model [model_name] [kwargs]
    parser_run = subparsers.add_parser("run", help="Run an eval suite")
    parser_run.add_argument(
        "eval_name",
        type=str,
        help="The name of the eval or eval suite to run",
    )
    parser_run.add_argument(
        "kwargs",
        nargs="*",
        action=KeyValueAction,
        help="Additional named arguments as key=value pairs",
    )
    parser_run.set_defaults(func=run)

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

    llm_engine = input("Which OpenAI model should we use for evals: ")
    config_data["llm_engine"] = llm_engine

    # Add other configuration prompts as needed

    ConfigHelper.save_config(config_data)
    print("Configuration updated successfully. See athina_config.yml for details.")


def config(args):
    """Prints the current configuration"""
    config_data = ConfigHelper.load_config()
    print(config_data)


def list(args):
    """Lists all available evals"""
    evals = RunHelper.all_evals()
    evals_list = "- "
    evals_list += "\n- ".join(evals)
    print(evals_list)


def run(args):
    """Runs a single eval on a single datapoint"""
    eval_name = args.eval_name
    model = ConfigHelper.load_llm_engine()
    kwargs = args.kwargs

    try:
        RunHelper.run_eval(eval_name, model, kwargs)
    except Exception as e:
        print(f"{e}")
        return


if __name__ == "__main__":
    main()
