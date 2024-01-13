import re
import json
import requests
from typing import Any

def _standardize_url(url):
    """
    Generate a standardized URL by adding 'http://' if it's missing.

    Args:
        url (str): The input URL to be standardized.

    Returns:
        str: The standardized URL.
    """
    if url.startswith("http://") or url.startswith("https://"):
        return url
    else:
        return "http://" + url

def regex(pattern, response):
    """
    Perform a regex search on the response and return a dictionary indicating whether the pattern was found.

    Args:
        pattern (str): The regex pattern to search for.
        response (str): The response string to search within.

    Returns:
        dict: A dictionary containing the result of the regex search and the reason for the result.
    """
    match = re.search(pattern, response)
    if match:
        return {"result": True, "reason": f"regex pattern {pattern} found in output"}
    else:
        return {
            "result": False,
            "reason": f"regex pattern {pattern} not found in output",
        }

def contains_any(keywords, response, case_sensitive=False):
    """
    Check if any of the provided keywords are present in the response.

    Args:
        keywords (str or List[str]): The keyword(s) to search for in the response.
        response (str): The response string to search within.
        case_sensitive (bool, optional): Whether the search should be case-sensitive. Defaults to False.

    Returns:
        dict: A dictionary containing the result of the search and the reason for the result.
    """
    if isinstance(keywords, str):
        keywords = keywords.split(",")
    keywords = list(map(lambda k: k.strip(), keywords))
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

def contains_all(keywords, response, case_sensitive=False):
    """
    Check if all the provided keywords are present in the response.

    Args:
        keywords (List[str]): The list of keywords to search for in the response.
        response (str): The response string to search within.
        case_sensitive (bool, optional): If True, the comparison is case-sensitive. Defaults to False.

    Returns:
        dict: A dictionary containing the result of the keyword search and the reason for the result.
    """
    if case_sensitive == False:
        response = response.lower()
        keywords = list(map(lambda k: k.lower(), keywords))
    missing_keywords = []
    for keyword in keywords:
        if keyword not in response:
            result = False
            missing_keywords.append(keyword)
    if (len(missing_keywords)) > 0:
        result = False
        reason = f"keywords not found in output: " + ", ".join(missing_keywords)
    else:
        result = True
        reason = f"{len(keywords)}/{len(keywords)} keywords found in output"

    return {"result": result, "reason": reason}

def contains(keyword, response, case_sensitive=False):
    """
    Check if the response contains a specific keyword.

    Args:
        keyword (str): The keyword to search for in the response.
        response (str): The response string to search within.
        case_sensitive (bool, optional): If True, the comparison is case-sensitive. Defaults to False.

    Returns:
        dict: A dictionary containing the result of the keyword search and the reason for the result.
    """
    if case_sensitive == False:
        response = response.lower()
        keyword = keyword.lower()
    if keyword not in response:
        result = False
        reason = f"keyword not found in output: " + keyword
    else:
        result = True
        reason = f"keyword {keyword} found in output"

    return {"result": result, "reason": reason}

def contains_none(keywords, response, case_sensitive=False):
    """
    Check if none of the provided keywords are present in the response.

    Args:
        keywords (str or List[str]): The keyword(s) to search for in the response.
        response (str): The response string to search within.
        case_sensitive (bool, optional): If True, the comparison is case-sensitive. Defaults to False.

    Returns:
        dict: A dictionary containing the result of the check and the reason for the result.
    """
    if not case_sensitive:
        response = response.lower()
        keywords = list(map(lambda k: k.lower(), keywords))

    found_keywords = []
    for keyword in keywords:
        if keyword in response:
            found_keywords.append(keyword)

    if found_keywords:
        result = False
        reason = f"One or more keywords were found in output: " + ", ".join(
            found_keywords
        )
    else:
        result = True
        reason = "No keywords found in output"

    return {"result": result, "reason": reason}

def contains_json(response):
    """
    Check if the response contains valid JSON.

    Args:
        response (str): The response string to check for valid JSON.

    Returns:
        dict: A dictionary containing the result of the JSON check and the reason for the result.
    """
    trimmed_output = response.strip()
    pattern = r'\{(?:\s*"(?:\\.|[^"\\])*"\s*:\s*(?:"(?:\\.|[^"\\])*"|[^{}\[\]:,]+)|[^{}]+)*\}'
    matches = re.findall(pattern, trimmed_output)

    if matches:
        results = []
        errors = []
        for potential_json_string in matches:
            print(potential_json_string)
            try:
                parsed_json = json.loads(potential_json_string)
                results.append({"json": parsed_json, "valid": True})
            except json.JSONDecodeError as e:
                errors.append({"json": potential_json_string, "valid": False, "error": str(e)})
        if errors:
            return {"result": False, "reason": "Output contains a potential JSON but it is invalid", "matches": results, "errors": errors}
        else:   
            return {"result": True, "reason": "Output contains JSON", "matches": results}
    else:
        return {"result": False, "reason": "Output does not contain JSON"}
    
def contains_email(response):
    """
    Check if the response contains an email address.

    Args:
        response (str): The response string to check for an email address.

    Returns:
        dict: A dictionary containing the result of the email address check and the reason for the result.
    """
    return regex(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', response)

def contains_phone_number(response):
    """
    Check if the response contains a phone number.

    Args:
        response (str): The response string to check for a phone number.

    Returns:
        dict: A dictionary containing the result of the phone number check and the reason for the result.
    """
    return regex(r'\+?1?\d{9,15}', response)

def is_json(response):
    """
    Check if the response contains valid JSON.

    Args:
        response (str): The response string to check for valid JSON.

    Returns:
        dict: A dictionary containing the result of the JSON check and the reason for the result.
    """
    try:
        json.loads(response)
        result = True
    except json.JSONDecodeError:
        result = False
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

def is_email(response):
    """
    Check if the response is a valid email address.

    Args:
        response (str): The response string to check for a valid email address.

    Returns:
        dict: A dictionary containing the result of the email address check and the reason for the result.
    """
    return regex(r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$', response)

def is_phone_number(response):
    """
    Check if the response is a valid phone number.

    Args:
        response (str): The response string to check for a valid phone number.

    Returns:
        dict: A dictionary containing the result of the phone number check and the reason for the result.
    """
    return regex(r'^\+?1?\d{9,15}$', response)

def contains_link(response):
    """
    Check if the response contains a link.

    Args:
        response (str): The response string to check for a link.

    Returns:
        dict: A dictionary containing the result of the link check and the reason for the result.
    """
    pattern = r"(?!.*@)(?:https?://)?(?:www\.)?\S+\.\S+"
    result = bool(re.search(pattern, response))
    if result:
        return {"result": True, "reason": "Link found in output"}
    else:
        return {"result": False, "reason": "No link found in output"}

def contains_valid_link(response):
    """
    Check if the response contains a valid link.

    Args:
        response (str): The response string to check for a valid link.

    Returns:
        dict: A dictionary containing the result of the link check and the reason for the result.
    """
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
    return {"result": False, "reason": f"no link found in output"}

def no_invalid_links(response):
    """
    Check for invalid links in the response.

    Args:
        response (str): The response string to check for invalid links.

    Returns:
        dict: A dictionary containing the result of the link check and the reason for the result.
    """
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
    return {"result": True, "reason": f"no invalid link found in output"}

def api_call(
    url: str,
    payload: dict,
    response,
    headers: dict = None,
):
    """
    Make an API call with payload to the specified URL.

    Args:
        url (str): The URL to make the API call to.
        payload (dict): The payload to be sent in the API call.
        response: The response to be added to the payload.
        headers (dict, optional): The headers to be included in the API call. Defaults to None.

    Returns:
        dict: A dictionary containing the result and reason of the API call.
    """
    payload["response"] = response
    api_response = requests.post(url, json=payload, headers=headers)
    # TODO: Add support to json path way of extracting result and reason
    result = api_response.json().get("result")
    reason = api_response.json().get("reason") 
    return {
        "result": result,
        "reason": reason,
    }

def equals(expected_output, response, case_sensitive=False):
    """
    Check if the response exactly matches the expected output.

    Args:
        expected_output (str): The expected output to compare against.
        response (str): The response to compare with the expected output.
        case_sensitive (bool, optional): If True, the comparison is case-sensitive. Defaults to False.

    Returns:
        dict: A dictionary containing the result and reason of the comparison.
    """
    if case_sensitive == False:
        response = response.lower()
        expected_output = expected_output.lower()
    if response == expected_output:
        result = True
        reason = "✅ output exactly matches expected output"
    else:
        result = False
        reason = "output does not exactly match expected output"
    return {"result": result, "reason": reason}

def starts_with(substring, response, case_sensitive=False):
    """
    Check if the response starts with a specified substring.

    Args:
        substring (str): The substring to check for at the start of the response.
        response (str): The response string to check.
        case_sensitive (bool, optional): If True, the comparison is case-sensitive. Defaults to False.

    Returns:
        dict: A dictionary containing the result of the check and the reason for the result.
    """
    if case_sensitive == False:
        response = response.lower()
        substring = substring.lower()
    result = response.startswith(substring)
    if result == True:
        return {"result": result, "reason": "output starts with " + substring}
    else:
        return {"result": result, "reason": "output does not start with " + substring}

def ends_with(substring, response, case_sensitive=False):
    """
    Check if the response ends with a specified substring.

    Args:
        substring (str): The substring to check for at the end of the response.
        response (str): The response string to check.
        case_sensitive (bool, optional): If True, the comparison is case-sensitive. Defaults to False.

    Returns:
        dict: A dictionary containing the result of the check and the reason for the result.
    """
    if case_sensitive == False:
        response = response.lower()
        substring = substring.lower()
    result = response.endswith(substring)
    if result == True:
        return {"result": result, "reason": "output ends with " + substring}
    else:
        return {"result": result, "reason": "output does not end with " + substring}

def length_less_than(max_length, response):
    """
    Check if the length of the response is less than a specified maximum length.

    Args:
        max_length (int): The maximum length that the response should have.
        response (str): The response string to check the length of.

    Returns:
        dict: A dictionary containing the result of the length check and the reason for the result.
    """
    if len(response) < max_length:
        return {
            "result": True,
            "reason": f"output length is less than {max_length} characters",
        }
    else:
        return {
            "result": False,
            "reason": f"output length is greater than {max_length} characters",
        }

def length_greater_than(min_length, response):
    """
    Check if the length of the response is greater than a specified minimum length.

    Args:
        min_length (int): The minimum length that the response should have.
        response (str): The response string to check the length of.

    Returns:
        dict: A dictionary containing the result of the length check and the reason for the result.
    """
    if len(response) > min_length:
        return {
            "result": True,
            "reason": f"output length is greater than {min_length} characters",
        }
    else:
        return {
            "result": False,
            "reason": f"output length is less than {min_length} characters",
        }

"""
A dictionary containing the available operations and their corresponding functions.
"""
operations = {
    "Regex": regex,
    "ContainsAny": contains_any,
    "ContainsAll": contains_all,
    "Contains": contains,
    "ContainsNone": contains_none,
    "ContainsJson": contains_json,
    "ContainsEmail": contains_email,
    "ContainsPhoneNumber": contains_phone_number,
    "IsJson": is_json,
    "IsEmail": is_email,
    "IsPhoneNumber": is_phone_number,
    "NoInvalidLinks": no_invalid_links,
    "ContainsLink": contains_link,
    "ContainsValidLink": contains_valid_link,
    "Equals": equals,
    "StartsWith": starts_with,
    "EndsWith": ends_with,
    "LengthLessThan": length_less_than,
    "LengthGreaterThan": length_greater_than,
    "ApiCall": api_call,
}