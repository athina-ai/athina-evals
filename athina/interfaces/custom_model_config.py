from pydantic import BaseModel
from typing import List, Dict, Any


class CustomModelConfig(BaseModel):
    completion_config: List[Dict[str, Any]]
    env_config: List[Dict[str, Any]]

"""
For azure, this config looks like this:
{
    "completion_config": [
    {    
        "api_base": "<YOUR_AZURE_DEPLOYMENT_API_BASE>"
    },
    {
         "api_version": "<YOUR_AZURE_DEPLOYMENT_API_VERSION>"
    }
    ],
    "env_config": []  
}
"""