from typing import List, Optional, Dict
from athina.interfaces.athina import AthinaFilters


class ConversationLoader():
    """
    This class is a data loader for conversation data

    Attributes:
        raw_dataset: The raw dataset as loaded from the source.
        processed_dataset: The processed dataset is the list of strings
    """

    def __init__(
        self,
    ):
        """
        Initializes the loader with specified or default column names.
        """
        self._raw_dataset = {}
        self._processed_dataset = []
    

    def load_athina_inferences(
        self,
        filters: Optional[AthinaFilters] = None,
        limit: int = 10,
        context_key: Optional[str] = None,
    ):
        """
        Load data from Athina API.
        """
        pass

    def load_from_string_array(self, strings: List[str]):
        """
        Loads data from a list of strings.

        :param strings: List of strings to be loaded.
        """
        if strings is None or not all(isinstance(s, str) for s in strings):
            raise ValueError("Input must be a list of strings")
        
        self._processed_dataset.extend(strings)
    
    def load_from_openai_messages(self, messages: List[List[Dict[str, str]]]):
        """
        Processes and loads data from an array of lists containing messages.

        :param messages: Array of lists of messages with roles and content.
        """
        if not all(isinstance(msg_list, list) for msg_list in messages):
            raise ValueError("Input must be an array of lists")

        for msg_list in messages:
            for msg in msg_list:
                if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                    raise ValueError("Each message must be a dict with 'role' and 'content' keys")
                prefix = "AI: " if msg['role'] == "assistant" else "User: "
                self._processed_dataset.append(prefix + msg['content'])
        