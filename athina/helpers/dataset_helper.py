import datetime

def generate_unique_dataset_name(prefix="Dataset-", separator="-"):
    """Generates a unique name using the current timestamp with separators for readability.
    
    Args:
        prefix (str): Optional. A prefix for the generated name.
        separator (str): The separator to use between date and time components.
    
    Returns:
        str: A unique name based on the current timestamp
    """
    # Get the current datetime with desired precision
    current_time = datetime.datetime.now()
    # Format the datetime into a string with separators
    time_str = current_time.strftime(f"%Y{separator}%m{separator}%d{separator}%H{separator}%M{separator}%S")
    # Combine the prefix and the formatted time string to create a unique name
    return prefix + time_str