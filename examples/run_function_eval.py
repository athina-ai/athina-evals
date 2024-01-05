import os
from dotenv import load_dotenv
load_dotenv()

from athina.evals.function.function_evaluator import FunctionEvaluator
from athina.loaders import FunctionEvalLoader
from athina.keys import AthinaApiKey, OpenAiApiKey

OpenAiApiKey.set_key(os.getenv('OPENAI_API_KEY'))
AthinaApiKey.set_key(os.getenv('ATHINA_API_KEY'))

raw_data = [
    { 
        "response": "New York",
    },
    { 
        "response": "Greece",
    }
]
function_arguments = {
    "keywords" : "Greece is often called the cradle of Western civilization.".split(' ')
}
function_name = "contains_any"
dataset = FunctionEvalLoader(function_name=function_name, function_arguments=function_arguments).load_dict(raw_data)
response = FunctionEvaluator(function_name=function_name, function_arguments=function_arguments).run_batch(data=dataset).to_df()
print(response)

raw_data = [
    { 
        "response": "If you're having any trouble setting it up, write to hello@athina.ai, and we'll be right there to help you out.",
    },
    { 
        "response": "If you're having any trouble setting it up, write to us, and we'll be right there to help you out.",
    }
]
function_arguments = {
    "pattern": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
}
function_name = "regex"
dataset = FunctionEvalLoader(function_name=function_name, function_arguments=function_arguments).load_dict(raw_data)
response = FunctionEvaluator(function_name=function_name, function_arguments=function_arguments).run_batch(data=dataset).to_df()
print(response)
