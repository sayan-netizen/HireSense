"""
Reusable Ollama Client for HireSense.
Handles HTTP communication, timeouts, error reporting, and model availability.
"""

import json
import urllib.request
import urllib.error
import socket
from typing import Dict, Any, List, Tuple, Optional
from config.settings import (
    OLLAMA_CHAT_ENDPOINT,
    OLLAMA_TAGS_ENDPOINT,
    DEFAULT_OLLAMA_MODEL,
    LLM_TIMEOUT_SECONDS,
    LLM_TEMPERATURE,
    LLM_NUM_PREDICT,
    LLM_NUM_CTX,
)


class OllamaClient:
    """Robust client for local Ollama HTTP API."""

    def __init__(
        self,
        chat_endpoint: str = OLLAMA_CHAT_ENDPOINT,
        tags_endpoint: str = OLLAMA_TAGS_ENDPOINT,
        default_model: str = DEFAULT_OLLAMA_MODEL,
        timeout: int = LLM_TIMEOUT_SECONDS,
    ):
        self.chat_endpoint = chat_endpoint
        self.tags_endpoint = tags_endpoint
        self.default_model = default_model
        self.timeout = timeout

    def check_connection(self) -> Tuple[bool, str, List[str]]:
        """
        Check if Ollama server is reachable and fetch installed models.
        Returns:
            (is_online, message, model_names)
        """
        try:
            req = urllib.request.Request(self.tags_endpoint, headers={"Accept": "application/json"}, method="GET")
            with urllib.request.urlopen(req, timeout=4) as response:
                if response.status == 200:
                    payload = json.loads(response.read().decode("utf-8"))
                    models = [m.get("name", "") for m in payload.get("models", [])]
                    return True, "Ollama server is active and reachable.", models
                return False, f"Ollama returned HTTP status {response.status}", []
        except urllib.error.URLError as e:
            return False, f"Cannot connect to Ollama (Connection Refused): {e.reason}", []
        except (TimeoutError, socket.timeout):
            return False, "Connection to Ollama timed out.", []
        except Exception as e:
            return False, f"Ollama connection error: {str(e)}", []

    def get_available_models(self) -> List[str]:
        """Fetch list of available models from Ollama."""
        _, _, models = self.check_connection()
        return models

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        json_format: bool = True,
        temperature: float = LLM_TEMPERATURE,
        num_predict: int = LLM_NUM_PREDICT,
        num_ctx: int = LLM_NUM_CTX,
    ) -> str:
        """
        Send a chat completion request to Ollama.
        Returns the raw message content string.
        """
        active_model = model or self.default_model
        payload: Dict[str, Any] = {
            "model": active_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "num_ctx": num_ctx,
            },
        }

        if json_format:
            payload["format"] = "json"
            payload["think"] = False

        req_data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.chat_endpoint,
            data=req_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            
            content = result.get("message", {}).get("content", "").strip()
            if not content:
                raise RuntimeError(f"Ollama model '{active_model}' returned an empty response.")
            return content

        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP error {e.code}: {error_body}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.chat_endpoint}. "
                "Ensure the Ollama application or service is running locally ('ollama serve')."
            ) from e
        except (TimeoutError, socket.timeout) as e:
            raise RuntimeError(
                f"Ollama took longer than {self.timeout}s to respond. "
                "Try shortening your answer or switching to a faster model."
            ) from e
        except Exception as e:
            raise RuntimeError(f"Ollama communication failed: {str(e)}") from e
