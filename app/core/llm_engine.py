"""Ollama LLM wrapper for Lullus.

Provides a unified interface to generate text using locally-running
Ollama models, with retry logic, streaming support, and model management.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import yaml

try:
    import ollama
except ImportError:
    ollama = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 1.0  # seconds


class LLMConnectionError(Exception):
    """Raised when Ollama is not reachable."""


class LLMGenerationError(Exception):
    """Raised when text generation fails after retries."""


class LLMEngine:
    """Wrapper around the Ollama Python client for text generation.

    Loads default configuration from config/default_config.yaml and
    provides generate, streaming, model listing, and health-check methods.

    Args:
        base_dir: Root directory of the Lullus project. Defaults to
            two levels above this file.
        model: Override the default model name.
        temperature: Override the default temperature.
        max_tokens: Override the default max_tokens.
        max_retries: Number of retries with exponential backoff.
        retry_base_delay: Base delay in seconds between retries.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    ) -> None:
        if ollama is None:
            raise ImportError(
                "The 'ollama' package is required. Install it with: pip install ollama"
            )

        if base_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
        self.base_dir = Path(base_dir)

        # Load default config
        config = self._load_config()
        llm_config: Dict[str, Any] = config.get("llm", {})

        self.model: str = model or llm_config.get("model", "mistral:7b-instruct-v0.3-q4_K_M")
        self.temperature: float = temperature if temperature is not None else llm_config.get("temperature", 0.7)
        self.max_tokens: int = max_tokens if max_tokens is not None else llm_config.get("max_tokens", 2048)
        self.context_window: int = llm_config.get("context_window", 8192)
        self.max_retries: int = max_retries
        self.retry_base_delay: float = retry_base_delay

        logger.info(
            "LLMEngine initialised: model=%s, temperature=%.2f, max_tokens=%d",
            self.model,
            self.temperature,
            self.max_tokens,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a complete text response.

        Args:
            prompt: The user prompt / question.
            system_prompt: Optional system-level instruction.
            temperature: Override instance temperature for this call.
            max_tokens: Override instance max_tokens for this call.

        Returns:
            The generated text.

        Raises:
            LLMConnectionError: If Ollama cannot be reached.
            LLMGenerationError: If generation fails after all retries.
        """
        messages = self._build_messages(prompt, system_prompt)
        options = self._build_options(temperature, max_tokens)

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug("generate() attempt %d/%d", attempt, self.max_retries)
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options=options,
                    stream=False,
                )
                content: str = response["message"]["content"]
                logger.info("generate() succeeded on attempt %d (%d chars)", attempt, len(content))
                return content

            except ollama.ResponseError as exc:
                last_error = exc
                logger.warning("Ollama response error (attempt %d): %s", attempt, exc)
                self._backoff(attempt)

            except Exception as exc:
                last_error = exc
                if self._is_connection_error(exc):
                    raise LLMConnectionError(
                        "Cannot connect to Ollama. Make sure it is running "
                        "(start with 'ollama serve') and accessible at http://localhost:11434."
                    ) from exc
                logger.warning("Unexpected error (attempt %d): %s", attempt, exc)
                self._backoff(attempt)

        raise LLMGenerationError(
            f"Text generation failed after {self.max_retries} retries. "
            f"Last error: {last_error}"
        )

    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Generate text as a stream of chunks.

        Args:
            prompt: The user prompt / question.
            system_prompt: Optional system-level instruction.
            temperature: Override instance temperature for this call.
            max_tokens: Override instance max_tokens for this call.

        Yields:
            String chunks as they are produced by the model.

        Raises:
            LLMConnectionError: If Ollama cannot be reached.
            LLMGenerationError: If generation fails after all retries.
        """
        messages = self._build_messages(prompt, system_prompt)
        options = self._build_options(temperature, max_tokens)

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug("generate_stream() attempt %d/%d", attempt, self.max_retries)
                stream = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options=options,
                    stream=True,
                )
                total_chars = 0
                for chunk in stream:
                    token: str = chunk["message"]["content"]
                    total_chars += len(token)
                    yield token
                logger.info("generate_stream() succeeded on attempt %d (%d chars)", attempt, total_chars)
                return  # Successful completion

            except ollama.ResponseError as exc:
                last_error = exc
                logger.warning("Ollama response error in stream (attempt %d): %s", attempt, exc)
                self._backoff(attempt)

            except Exception as exc:
                last_error = exc
                if self._is_connection_error(exc):
                    raise LLMConnectionError(
                        "Cannot connect to Ollama. Make sure it is running "
                        "(start with 'ollama serve') and accessible at http://localhost:11434."
                    ) from exc
                logger.warning("Unexpected error in stream (attempt %d): %s", attempt, exc)
                self._backoff(attempt)

        raise LLMGenerationError(
            f"Streaming generation failed after {self.max_retries} retries. "
            f"Last error: {last_error}"
        )

    def list_models(self) -> List[str]:
        """List models available in the local Ollama instance.

        Returns:
            A list of model name strings.

        Raises:
            LLMConnectionError: If Ollama cannot be reached.
        """
        try:
            response = ollama.list()
            models: List[str] = [m["name"] for m in response.get("models", [])]
            logger.info("Found %d local models", len(models))
            return models
        except Exception as exc:
            if self._is_connection_error(exc):
                raise LLMConnectionError(
                    "Cannot connect to Ollama. Make sure it is running "
                    "(start with 'ollama serve') and accessible at http://localhost:11434."
                ) from exc
            logger.error("Error listing models: %s", exc)
            raise

    def check_connection(self) -> bool:
        """Check whether Ollama is reachable.

        Returns:
            True if Ollama responds, False otherwise.
        """
        try:
            ollama.list()
            logger.debug("Ollama connection check: OK")
            return True
        except Exception as exc:
            logger.warning("Ollama connection check failed: %s", exc)
            return False

    def pull_model(self, model_name: str) -> bool:
        """Pull (download) a model into the local Ollama instance.

        Args:
            model_name: The model tag to pull (e.g. 'mistral:7b').

        Returns:
            True if the pull succeeded, False otherwise.
        """
        try:
            logger.info("Pulling model: %s (this may take a while)...", model_name)
            ollama.pull(model_name)
            logger.info("Model pulled successfully: %s", model_name)
            return True
        except Exception as exc:
            if self._is_connection_error(exc):
                logger.error(
                    "Cannot connect to Ollama to pull model. "
                    "Make sure Ollama is running."
                )
            else:
                logger.error("Failed to pull model '%s': %s", model_name, exc)
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_config(self) -> Dict[str, Any]:
        """Load the default configuration YAML."""
        config_path = self.base_dir / "config" / "default_config.yaml"
        if not config_path.exists():
            logger.warning("Config file not found at %s, using built-in defaults", config_path)
            return {}
        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                config: Dict[str, Any] = yaml.safe_load(fh) or {}
            logger.debug("Config loaded from %s", config_path)
            return config
        except Exception as exc:
            logger.warning("Error reading config file: %s, using built-in defaults", exc)
            return {}

    @staticmethod
    def _build_messages(
        prompt: str, system_prompt: Optional[str]
    ) -> List[Dict[str, str]]:
        """Build the messages list for ollama.chat()."""
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_options(
        self,
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> Dict[str, Any]:
        """Build the options dict for ollama.chat()."""
        return {
            "temperature": temperature if temperature is not None else self.temperature,
            "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
        }

    def _backoff(self, attempt: int) -> None:
        """Sleep with exponential backoff."""
        if attempt < self.max_retries:
            delay = self.retry_base_delay * (2 ** (attempt - 1))
            logger.debug("Backing off %.1f seconds before retry", delay)
            time.sleep(delay)

    @staticmethod
    def _is_connection_error(exc: Exception) -> bool:
        """Heuristically detect connection-related errors."""
        error_str = str(exc).lower()
        connection_indicators = [
            "connection refused",
            "connect call failed",
            "no route to host",
            "connection reset",
            "name or service not known",
            "errno 111",
            "errno 61",
            "failed to connect",
        ]
        return any(indicator in error_str for indicator in connection_indicators)
