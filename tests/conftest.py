from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List

import httpx
import pytest
from openai import RateLimitError

from backend.core.api_manager import APIKeyState, APIManager, ProviderConfig


class SequencedChatCompletions:
    """OpenAI-compatible chat completion stub that returns configured outcomes."""

    def __init__(self, outcomes: List[Any]) -> None:
        self.outcomes = outcomes
        self.calls = 0

    def create(self, **kwargs: Any) -> Any:
        """Return or raise the next configured outcome."""
        self.calls += 1
        if not self.outcomes:
            raise AssertionError("No mocked API outcome remains.")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class SequencedOpenAIClient:
    """Minimal OpenAI client stub exposing chat.completions.create."""

    def __init__(self, outcomes: List[Any]) -> None:
        self.chat = SimpleNamespace(completions=SequencedChatCompletions(outcomes))


@pytest.fixture
def provider_config() -> ProviderConfig:
    """Return a two-key Moonshot provider pool with low quotas for deterministic tests."""
    return ProviderConfig(
        provider="moonshot",
        base_url="https://api.moonshot.cn/v1",
        model="kimi-k2.6",
        key_env_var="MOONSHOT_KEYS",
        quota_env_var="MOONSHOT_SAFE_TOKEN_QUOTA",
        keys=[
            APIKeyState(key="key-alpha-0001", safe_quota_tokens=100),
            APIKeyState(key="key-bravo-0002", safe_quota_tokens=100),
        ],
    )


@pytest.fixture
def api_manager(provider_config: ProviderConfig) -> APIManager:
    """Return an API manager wired only to mocked provider state."""
    return APIManager(provider_configs=[provider_config])


def mocked_response(content: str = "mocked answer", total_tokens: int = 12) -> Any:
    """Build an OpenAI-compatible response object without making network calls."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(total_tokens=total_tokens),
    )


def mocked_rate_limit_error() -> RateLimitError:
    """Build an OpenAI-compatible 429 quota error for retry tests."""
    request = httpx.Request("POST", "https://api.moonshot.cn/v1/chat/completions")
    response = httpx.Response(
        429,
        request=request,
        json={"error": {"code": "insufficient_quota", "message": "quota exhausted"}},
    )
    return RateLimitError(
        "insufficient_quota",
        response=response,
        body={"error": {"code": "insufficient_quota", "message": "quota exhausted"}},
    )
