import os

from dotenv import load_dotenv
load_dotenv()

from athina.evals.function.function_evaluator import FunctionEvaluator
from athina.loaders import FunctionEvalLoader
from athina.keys import AthinaApiKey, OpenAiApiKey

OpenAiApiKey.set_key(os.getenv('OPENAI_API_KEY'))
AthinaApiKey.set_key(os.getenv('ATHINA_API_KEY'))

keywords = "Greece is often called the cradle of Western civilization.".split(' ')
raw_data = [
    { 
        "response": "Delhi",
    },
    { 
        "response": "Greece",
    }
]

function_name = "contains_any"
dataset = FunctionEvalLoader(function_name=function_name).load_dict(raw_data)
response = FunctionEvaluator(function_name=function_name, function_args=).run_batch(data=dataset).to_df()
print(response)

raw_data = [
    { 
        "pattern": r"^\S+@\S+\.\S+$",
        "response": "If you're having any trouble setting it up, write to hello@athina.ai, and we'll be right there to help you out.",
    }
]

function_name = "regex"
dataset = FunctionEvalLoader(function_name=function_name).load_dict(raw_data)
response = FunctionEvaluator(function_name=function_name).run_batch(data=dataset).to_df()
print(response)


"""
# TODO
Create FunctionEvaluator
Add a constructor (should support custom FunctionEvaluator)
Ideally, this could be as simple as a constructor function which takes in properties like name, display name, eval_function)
Library of simple function evaluators (like contains, regex_match, api_request, etc)
Add job type to eval_type table
"""