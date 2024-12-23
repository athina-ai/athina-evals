from typing import Union, Dict, Any, Optional
from athina.steps import Step
from llama_parse import LlamaParse
import nest_asyncio

nest_asyncio.apply() # LlamaParse can cause nested asyncio exceptions so we need this line of code

class ParseDocument(Step):
    """
    Step that uses the llama_parse package to extract text from various document formats.

    Attributes:
        file_url: The URL of the file to be parsed.
        output_format: The type of result to return. Options: 'text' or 'markdown'. Default is 'text'.
        llama_parse_key: The API key to use for the LlamaParse API.
        verbose: Whether to print verbose output. Default is False.
    """

    file_url: str
    output_format: Optional[str] = "text"
    llama_parse_key: str
    verbose: Optional[bool] = False

    def execute(self, input_data) -> Union[Dict[str, Any], None]:
        """Parse a document using LlamaParse and return the result."""

        if input_data is None:
            input_data = {}

        if not isinstance(input_data, dict):
            raise TypeError("Input data must be a dictionary.")

        try:
            # Initialize LlamaParse client
            llama_parse = LlamaParse(api_key=self.llama_parse_key, verbose=self.verbose, result_type=self.output_format)

            # Parse the document
            documents = llama_parse.load_data(
                file_path=self.file_url
            )
            
            if not documents:
                return {
                    "status": "error",
                    "data": "No documents were parsed."
                }

            parsed_content = documents[0].text

            return {
                "status": "success",
                "data": parsed_content,
            }

        except Exception as e:
            return {
                "status": "error",
                "data": f"LlamaParse error: {str(e)}",
            }