"""Thread-safe API key firewall for OpenAI-compatible providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
import json
import os
import threading
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, AuthenticationError, OpenAI, RateLimitError
import tiktoken

from backend.core.logger import get_logger

LOGGER = get_logger(__name__)
DEFAULT_SAFE_TOKEN_QUOTA = 1_000_000
DEFAULT_SOFT_CAP_RATIO = 0.80
DEFAULT_MODEL = "kimi-k2.6"
DEFAULT_BASE_URL = "https://api.moonshot.cn/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_TRANSCRIPTION_MODEL = "gpt-4o-mini-transcribe"


class APIKeyPoolExhaustedError(RuntimeError):
    """Raised when every key in a provider pool is exhausted or unavailable."""


@dataclass
class APIKeyState:
    """Mutable accounting state for one API key."""

    key: str
    safe_quota_tokens: int = DEFAULT_SAFE_TOKEN_QUOTA
    used_tokens: int = 0
    status: str = "active"
    failures: int = 0
    last_error: str = ""

    @property
    def soft_cap_tokens(self) -> int:
        """Return the 80 percent soft-cap threshold for this key."""
        return int(self.safe_quota_tokens * DEFAULT_SOFT_CAP_RATIO)

    def can_spend(self, estimated_tokens: int) -> bool:
        """Return whether this key can safely spend the estimated tokens."""
        return self.status == "active" and (self.used_tokens + estimated_tokens) < self.soft_cap_tokens


@dataclass
class ProviderConfig:
    """Configuration for an OpenAI-compatible provider."""

    provider: str
    base_url: str
    model: str
    key_env_var: str
    quota_env_var: str
    keys: List[APIKeyState] = field(default_factory=list)


class APIManager:
    """Manage provider keys, token accounting, rotation, and retry behavior."""

    def __init__(self, provider_configs: Optional[Sequence[ProviderConfig]] = None) -> None:
        """Initialize the manager from explicit configs or environment variables.

        Args:
            provider_configs: Optional provider configurations for tests or custom runtime wiring.
        """
        load_dotenv()
        self._lock = threading.Lock()
        self._encoding_cache: Dict[str, Any] = {}
        self._tokenizer_available = True
        self._tokenizer_timeout_seconds = 2.0
        self._provider_configs: Dict[str, ProviderConfig] = {}
        self._client_factory: Callable[[str, str], OpenAI] = lambda api_key, base_url: OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        configs = list(provider_configs) if provider_configs is not None else [
            self._load_moonshot_config(),
            self._load_openai_config(),
        ]
        for config in configs:
            self._provider_configs[config.provider] = config

    def _load_moonshot_config(self) -> ProviderConfig:
        """Load the Moonshot/Kimi provider config from environment variables."""
        keys = parse_key_list(os.getenv("MOONSHOT_KEYS") or os.getenv("KIMI_KEYS") or os.getenv("OPENROUTER_KEYS") or "")
        single_key = os.getenv("MOONSHOT_API_KEY") or os.getenv("KIMI_API_KEY")
        if single_key and single_key not in keys:
            keys.append(single_key)
        quota = parse_int(os.getenv("MOONSHOT_SAFE_TOKEN_QUOTA"), DEFAULT_SAFE_TOKEN_QUOTA)
        states = [APIKeyState(key=key, safe_quota_tokens=quota) for key in keys]
        return ProviderConfig(
            provider="moonshot",
            base_url=os.getenv("MOONSHOT_BASE_URL", DEFAULT_BASE_URL),
            model=os.getenv("MOONSHOT_MODEL", DEFAULT_MODEL),
            key_env_var="MOONSHOT_KEYS",
            quota_env_var="MOONSHOT_SAFE_TOKEN_QUOTA",
            keys=states,
        )

    def _load_openai_config(self) -> ProviderConfig:
        """Load OpenAI provider config for speech transcription when configured."""
        keys = parse_key_list(os.getenv("OPENAI_KEYS") or "")
        single_key = os.getenv("OPENAI_API_KEY")
        if single_key and single_key not in keys:
            keys.append(single_key)
        quota = parse_int(os.getenv("OPENAI_SAFE_TOKEN_QUOTA"), DEFAULT_SAFE_TOKEN_QUOTA)
        states = [APIKeyState(key=key, safe_quota_tokens=quota) for key in keys]
        return ProviderConfig(
            provider="openai",
            base_url=os.getenv("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL),
            model=os.getenv("OPENAI_TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL),
            key_env_var="OPENAI_KEYS",
            quota_env_var="OPENAI_SAFE_TOKEN_QUOTA",
            keys=states,
        )

    def set_client_factory(self, factory: Callable[[str, str], OpenAI]) -> None:
        """Replace the client factory for tests or controlled dependency injection.

        Args:
            factory: Callable receiving API key and base URL and returning an OpenAI-compatible client.
        """
        self._client_factory = factory

    def estimate_tokens(self, messages: Sequence[Mapping[str, Any]], model: str = DEFAULT_MODEL) -> int:
        """Estimate request tokens using tiktoken with an OpenAI-compatible fallback.

        Args:
            messages: Chat messages that will be submitted to a provider.
            model: Target model name.

        Returns:
            Estimated input token count plus a small protocol overhead.
        """
        total = 0
        for message in messages:
            total += 4
            content = message.get("content", "")
            if isinstance(content, str):
                total += self._count_text_tokens(content, model)
            elif isinstance(content, list):
                total += self._count_text_tokens(json.dumps(content, ensure_ascii=False), model)
            else:
                total += self._count_text_tokens(str(content), model)
        return total + 4

    def _count_text_tokens(self, text: str, model: str) -> int:
        """Count text tokens with tiktoken and use a deterministic fallback if unavailable."""
        encoding = self._load_encoding(model)
        if encoding is not None:
            return len(encoding.encode(text))
        return max(1, (len(text) + 3) // 4)

    def _load_encoding(self, model: str) -> Optional[Any]:
        """Load a tiktoken encoding without allowing tokenizer startup to block requests."""
        if model in self._encoding_cache:
            return self._encoding_cache[model]
        if not self._tokenizer_available:
            return None
        result: Dict[str, Any] = {}

        def load() -> None:
            try:
                result["encoding"] = tiktoken.encoding_for_model(model)
            except KeyError:
                result["encoding"] = tiktoken.get_encoding("o200k_base")
            except Exception as exc:
                result["error"] = exc

        worker = threading.Thread(target=load, daemon=True)
        worker.start()
        worker.join(self._tokenizer_timeout_seconds)
        if worker.is_alive():
            self._tokenizer_available = False
            LOGGER.critical("tiktoken startup exceeded timeout; using conservative heuristic token estimates.")
            return None
        if "error" in result:
            self._tokenizer_available = False
            LOGGER.critical("tiktoken failed to initialize; using conservative heuristic token estimates: %s", result["error"])
            return None
        encoding = result.get("encoding")
        self._encoding_cache[model] = encoding
        return encoding

    def provider_status(self, provider: str = "moonshot") -> List[Dict[str, Any]]:
        """Return redacted provider key state for diagnostics.

        Args:
            provider: Provider identifier.

        Returns:
            List of redacted key status dictionaries.
        """
        config = self._get_provider(provider)
        with self._lock:
            return [
                {
                    "key": redact_key(state.key),
                    "status": state.status,
                    "used_tokens": state.used_tokens,
                    "safe_quota_tokens": state.safe_quota_tokens,
                    "soft_cap_tokens": state.soft_cap_tokens,
                    "failures": state.failures,
                    "last_error": state.last_error,
                }
                for state in config.keys
            ]

    def chat_completion(
        self,
        messages: Sequence[Mapping[str, Any]],
        provider: str = "moonshot",
        model: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        extra_body: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Call an OpenAI-compatible chat completion endpoint with protected key rotation.

        Args:
            messages: Chat messages for the model.
            provider: Provider identifier.
            model: Optional model override.
            max_tokens: Maximum output tokens requested.
            temperature: Sampling temperature.
            extra_body: Provider-specific request body fields.

        Returns:
            Assistant text content.

        Raises:
            APIKeyPoolExhaustedError: If no usable key remains.
            APIError: If a non-quota provider error occurs.
        """
        config = self._get_provider(provider)
        selected_model = model or config.model
        estimated_tokens = self.estimate_tokens(messages, selected_model) + max_tokens
        last_error: Optional[BaseException] = None
        attempted_keys: set[str] = set()

        while True:
            state = self._select_key(config, estimated_tokens, attempted_keys)
            attempted_keys.add(state.key)
            client = self._client_factory(state.key, config.base_url)
            try:
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=list(messages),
                    max_tokens=max_tokens,
                    temperature=temperature,
                    extra_body=dict(extra_body or {}),
                )
                content = response.choices[0].message.content or ""
                actual_tokens = extract_usage_tokens(response, estimated_tokens)
                self._record_success(state, actual_tokens)
                return content
            except (RateLimitError, AuthenticationError, APIConnectionError, APIError) as exc:
                last_error = exc
                if is_exhaustion_error(exc):
                    self._mark_exhausted(state, exc)
                    if len(attempted_keys) >= len(config.keys):
                        raise APIKeyPoolExhaustedError("All API keys are exhausted or rate-limited.") from exc
                    continue
                raise

            if len(attempted_keys) >= len(config.keys):
                break

        raise APIKeyPoolExhaustedError("No API key could complete the request.") from last_error

    def audio_transcription(
        self,
        audio_bytes: bytes,
        filename: str = "interview.wav",
        provider: str = "openai",
        model: Optional[str] = None,
    ) -> str:
        """Transcribe audio through an OpenAI-compatible transcription endpoint with key rotation."""
        config = self._get_provider(provider)
        selected_model = model or config.model
        estimated_tokens = max(256, len(audio_bytes) // 4)
        last_error: Optional[BaseException] = None
        attempted_keys: set[str] = set()

        while True:
            state = self._select_key(config, estimated_tokens, attempted_keys)
            attempted_keys.add(state.key)
            client = self._client_factory(state.key, config.base_url)
            try:
                audio_file = BytesIO(audio_bytes)
                audio_file.name = filename
                response = client.audio.transcriptions.create(model=selected_model, file=audio_file)
                transcript = getattr(response, "text", "")
                self._record_success(state, estimated_tokens)
                return str(transcript).strip()
            except (RateLimitError, AuthenticationError, APIConnectionError, APIError) as exc:
                last_error = exc
                if is_exhaustion_error(exc):
                    self._mark_exhausted(state, exc)
                    if len(attempted_keys) >= len(config.keys):
                        raise APIKeyPoolExhaustedError("All transcription API keys are exhausted or rate-limited.") from exc
                    continue
                raise

            if len(attempted_keys) >= len(config.keys):
                break

        raise APIKeyPoolExhaustedError("No API key could transcribe the audio.") from last_error

    def _get_provider(self, provider: str) -> ProviderConfig:
        """Return provider config or raise a typed pool error."""
        config = self._provider_configs.get(provider)
        if config is None:
            raise APIKeyPoolExhaustedError(f"Unknown API provider: {provider}")
        if not config.keys:
            raise APIKeyPoolExhaustedError(f"No API keys configured for provider: {provider}")
        return config

    def _select_key(self, config: ProviderConfig, estimated_tokens: int, attempted_keys: set[str]) -> APIKeyState:
        """Select the next active key that remains under its soft cap."""
        with self._lock:
            for state in config.keys:
                if state.status == "active" and not state.can_spend(estimated_tokens):
                    state.status = "soft_cap_reached"
                    state.last_error = "80 percent soft token cap reached"
                    LOGGER.critical("API key %s rotated at 80%% soft token cap.", redact_key(state.key))
            for state in config.keys:
                if state.key not in attempted_keys and state.can_spend(estimated_tokens):
                    return state
        raise APIKeyPoolExhaustedError("No active API key remains below the configured soft cap.")

    def _record_success(self, state: APIKeyState, tokens: int) -> None:
        """Record token usage and rotate the key if it crossed its soft cap."""
        with self._lock:
            state.used_tokens += max(tokens, 0)
            if state.used_tokens >= state.soft_cap_tokens:
                state.status = "soft_cap_reached"
                state.last_error = "80 percent soft token cap reached after request"
                LOGGER.critical("API key %s reached soft token cap after request.", redact_key(state.key))

    def _mark_exhausted(self, state: APIKeyState, exc: BaseException) -> None:
        """Mark a key as exhausted or unusable after an API quota/rate failure."""
        with self._lock:
            state.status = "exhausted"
            state.failures += 1
            state.last_error = str(exc)
            LOGGER.critical("API key %s marked exhausted after provider error: %s", redact_key(state.key), exc)


def parse_key_list(raw_value: str) -> List[str]:
    """Parse API key arrays stored as JSON, CSV, or semicolon-delimited strings."""
    value = raw_value.strip()
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        return [part.strip().strip('"').strip("'") for part in value.replace(";", ",").split(",") if part.strip()]
    return []


def parse_int(raw_value: Optional[str], default: int) -> int:
    """Parse a positive integer environment setting with a safe default."""
    if raw_value is None:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def redact_key(key: str) -> str:
    """Return a redacted API key suitable for logs and UI diagnostics."""
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}****{key[-4:]}"


def extract_usage_tokens(response: Any, fallback: int) -> int:
    """Extract provider usage tokens from an OpenAI-compatible response object."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return fallback
    total_tokens = getattr(usage, "total_tokens", None)
    if isinstance(total_tokens, int):
        return total_tokens
    if isinstance(usage, Mapping):
        value = usage.get("total_tokens")
        if isinstance(value, int):
            return value
    return fallback


def is_exhaustion_error(exc: BaseException) -> bool:
    """Return whether an exception should force key retirement and retry."""
    status_code = getattr(exc, "status_code", None)
    body = str(getattr(exc, "body", "")) + " " + str(exc)
    lowered = body.lower()
    return status_code in {402, 429} or "insufficient_quota" in lowered or "payment required" in lowered
