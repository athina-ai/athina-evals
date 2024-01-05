import re
import requests
from typing import Any

def _standardize_url(url):
    if url.startswith("http://") or url.startswith("https://"):
        return url
    else:
        return "http://" + url
    
def regex(pattern, response=None):
    match = re.search(pattern, response)
    if match:
        return {"result": True, "reason": f"regex pattern {pattern} found in output"}
    else:
        return {
            "result": False,
            "reason": f"regex pattern {pattern} not found in output",
        }

def contains_any(keywords, response=None, case_sensitive=False):
    if not case_sensitive:
        response = response.lower()
        keywords = list(map(lambda k: k.lower(), keywords))

    found_keywords = []
    for keyword in keywords:
        if keyword in response:
            found_keywords.append(keyword)

    if found_keywords:
        result = True
        reason = f"One or more keywords were found in output: " + ", ".join(
            found_keywords
        )
    else:
        result = False
        reason = "No keywords found in output"

    return {"result": result, "reason": reason}

def contains_json(response=None):
    trimmed_output = response.strip()
    pattern = r"^\{.*\}$|^\[.*\]$"
    result = bool(re.search(pattern, trimmed_output, re.DOTALL))
    if result:
        return {
            "result": True,
            "reason": "Output contains JSON",
        }
    else:
        return {
            "result": False,
            "reason": "Output does not contain JSON",
        }

def no_invalid_links(response=None):
    pattern = r"(?!.*@)(?:https?://)?(?:www\.)?\S+\.\S+"
    link_match = re.search(pattern=pattern, string=response)
    if link_match:
        matched_url = link_match.group()
        if matched_url:
            standardized_url = _standardize_url(matched_url)
            try:
                response = requests.head(standardized_url)
                if response.status_code == 200:
                    return {
                        "result": True,
                        "reason": f"link {matched_url} found in output and is valid",
                    }
                else:
                    return {
                        "result": False,
                        "reason": f"link {matched_url} found in output but is invalid",
                    }
            except:
                return {
                    "result": False,
                    "reason": f"link {matched_url} found in output but is invalid",
                }
    return {"result": True, "reason": f"no link found in output"}

def api_call(
    url: str,
    payload: dict,
    response = None,
    headers: dict = None,
):
    payload["response"] = response
    api_response = requests.post(url, json=payload, headers=headers)
    result = api_response.json().get("result")
    reason = api_response.json().get("reason")
    return {
        "result": result,
        "reason": reason,
    }

operations = {
    "regex": regex,
    "contains_any": contains_any,
    "contains_json": contains_json,
    "no_invalid_links": no_invalid_links,
}
