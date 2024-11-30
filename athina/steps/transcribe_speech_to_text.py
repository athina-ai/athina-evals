from typing import Any, Dict, Optional
import requests
from athina.steps import Step


class TranscribeSpeechToText(Step):
    """
    Step that transcribes audio to text using specified model.

    Attributes:
        audio_url: URL of the audio file to transcribe
        language: Language of the audio (optional)
        model: Model to use for transcription
        api_key: Deepgram API key
        profanity_filter: Remove profanity from transcript
        punctuate: Add punctuation and capitalization
        redact: Redact sensitive information
        replace: Terms to replace
        search: Terms to search for
        detect_language: Detect audio language
        filler_words: Include filler words
        diarize: Enable speaker diarization
        dictation: Convert spoken punctuation commands
    """

    audio_url: str
    language: Optional[str] = "en"
    model: str
    api_key: str
    profanity_filter: bool = False
    punctuate: bool = False
    redact: Optional[str] = None
    replace: Optional[str] = None
    search: Optional[str] = None
    detect_language: bool = False
    filler_words: bool = False
    diarize: bool = False
    dictation: bool = False

    def execute(self, input_data: Any) -> Dict[str, Any]:
        """Transcribe audio file and return the text."""
        try:
            # Prepare the request to Deepgram API
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "application/json",
            }

            payload = {"url": self.audio_url}

            # Build URL parameters
            params = {}
            if self.model is not None:
                params["model"] = self.model
            if self.language is not None:
                params["language"] = self.language

            # Add new parameters
            if self.profanity_filter:
                params["profanity_filter"] = "true"
            if self.punctuate:
                params["punctuate"] = "true"
            if self.redact:
                for item in self.redact.split(","):
                    params["redact"] = item.strip()
            if self.replace:
                for replacement in self.replace.split(","):
                    params["replace"] = replacement.strip()
            if self.search:
                for term in self.search.split(","):
                    params["search"] = term.strip()
            if self.detect_language:
                params["detect_language"] = "true"
            if self.filler_words:
                params["filler_words"] = "true"
            if self.diarize:
                params["diarize"] = "true"
            if self.dictation:
                params["dictation"] = "true"

            # Make request to Deepgram API
            response = requests.post(
                "https://api.deepgram.com/v1/listen",
                headers=headers,
                json=payload,
                params=params,
                timeout=30,
            )
            response.raise_for_status()

            # Parse the response
            result = response.json()

            # Extract the transcript
            transcribed_text = (
                result.get("results", {})
                .get("channels", [{}])[0]
                .get("alternatives", [{}])[0]
                .get("transcript", "")
            )

            # Create a simplified metadata object that's JSON serializable
            metadata = {
                "duration": result.get("metadata", {}).get("duration"),
                "channels": result.get("metadata", {}).get("channels"),
                "model": result.get("metadata", {}).get("model"),
                "language": result.get("metadata", {}).get("language"),
            }

            return {
                "status": "success",
                "data": transcribed_text,
                "metadata": metadata,  # Only include serializable metadata
            }

        except requests.RequestException as e:
            return {
                "status": "error",
                "data": f"Failed to download audio file: {str(e)}",
            }
        except Exception as e:
            return {
                "status": "error",
                "data": f"Transcription failed: {str(e)}",
            }
