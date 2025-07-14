"""Basic wrapper for interacting with an OLAMA model API."""

from typing import Optional
import requests


class LLM:
    """Helper for sending prompts to an OLAMA model."""

    #: Base URL for the OLAMA API. Can be overridden via configuration.
    api_url: str = "http://localhost:11434"

    #: Name of the model to query. Can be overridden via configuration.
    model_name: str = "default"

    def __init__(self, api_url: Optional[str] = None, model_name: Optional[str] = None) -> None:
        if api_url is not None:
            self.api_url = api_url
        if model_name is not None:
            self.model_name = model_name

    def process(self, prompt: str) -> str:
        """Send ``prompt`` to the configured OLAMA model and return the response."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        response = requests.post(f"{self.api_url}/api/generate", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        if "response" not in data:
            raise RuntimeError("Unexpected response from OLAMA API")
        return data["response"]
