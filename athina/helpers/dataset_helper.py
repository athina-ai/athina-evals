import datetime

def generate_unique_dataset_name(prefix="Dataset_"):
    """Generates a unique name using the current timestamp.
    
    Args:
    prefix (str): Optional. A prefix for the generated name.
    
    Returns:
    str: A unique name based on the current timestamp.
    """
    # Get the current datetime with microsecond precision
    current_time = datetime.datetime.now()
    # Format the datetime into a string with no spaces or special characters
    time_str = current_time.strftime("%Y%m%d%H%M%S%f")
    # Combine the prefix and the formatted time string to create a unique name
    return prefix + time_str