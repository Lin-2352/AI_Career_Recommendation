from __future__ import annotations

from unittest.mock import patch

import pytest

from backend.core.api_manager import APIKeyPoolExhaustedError, load_application_environment, parse_key_list
from tests.conftest import SequencedOpenAIClient, mocked_rate_limit_error, mocked_response


def test_parse_key_list_accepts_json_and_delimited_values() -> None:
    """API keys can be loaded from JSON arrays, comma lists, and semicolon lists."""
    assert parse_key_list('["one", "two"]') == ["one", "two"]
    assert parse_key_list("one,two;three") == ["one", "two", "three"]
    assert parse_key_list("") == []


def test_label_style_env_lines_are_loaded_without_exposing_values(tmp_path) -> None:
    """Label-style local key files are mapped to provider-specific key pools."""
    env_path = tmp_path / ".env"
    env_path.write_text(
        "open router key 1: router-secret\n"
        "fireworks ai api 1: fireworks-secret\n"
        "NVIDIA_NIM_API_KEYS=[\"nvidia-secret\"]\n",
        encoding="utf-8",
    )

    values, labeled = load_application_environment(env_path)

    assert values["NVIDIA_NIM_API_KEYS"] == '["nvidia-secret"]'
    assert [item.label for item in labeled] == ["open router key 1", "fireworks ai api 1"]


@patch("backend.core.api_manager.OpenAI")
def test_chat_completion_rotates_after_mocked_429(openai_factory, api_manager) -> None:
    """A mocked 429 retires the first key and retries with the next key."""
    clients = {
        "key-alpha-0001": SequencedOpenAIClient([mocked_rate_limit_error()]),
        "key-bravo-0002": SequencedOpenAIClient([mocked_response("rotated", total_tokens=17)]),
    }
    openai_factory.side_effect = lambda api_key, base_url: clients[api_key]

    result = api_manager.chat_completion(
        messages=[{"role": "user", "content": "career plan"}],
        max_tokens=8,
    )

    assert result == "rotated"
    status = api_manager.provider_status()
    assert status[0]["status"] == "exhausted"
    assert status[1]["used_tokens"] == 17
    assert clients["key-alpha-0001"].chat.completions.calls == 1
    assert clients["key-bravo-0002"].chat.completions.calls == 1


@patch("backend.core.api_manager.OpenAI")
def test_soft_cap_rotates_before_request(openai_factory, api_manager) -> None:
    """A key at the 80 percent soft cap is skipped before a live request can happen."""
    config = api_manager._get_provider("moonshot")
    config.keys[0].used_tokens = 79
    clients = {
        "key-alpha-0001": SequencedOpenAIClient([mocked_response("should not run")]),
        "key-bravo-0002": SequencedOpenAIClient([mocked_response("safe key", total_tokens=10)]),
    }
    openai_factory.side_effect = lambda api_key, base_url: clients[api_key]
    api_manager.estimate_tokens = lambda messages, model: 5

    result = api_manager.chat_completion(
        messages=[{"role": "user", "content": "small prompt"}],
        max_tokens=1,
    )

    assert result == "safe key"
    status = api_manager.provider_status()
    assert status[0]["status"] == "soft_cap_reached"
    assert status[1]["used_tokens"] == 10
    assert clients["key-alpha-0001"].chat.completions.calls == 0


@patch("backend.core.api_manager.OpenAI")
def test_all_keys_exhausted_raises_typed_error(openai_factory, api_manager) -> None:
    """When every mocked key returns quota failure, the manager raises a typed pool error."""
    clients = {
        "key-alpha-0001": SequencedOpenAIClient([mocked_rate_limit_error()]),
        "key-bravo-0002": SequencedOpenAIClient([mocked_rate_limit_error()]),
    }
    openai_factory.side_effect = lambda api_key, base_url: clients[api_key]

    with pytest.raises(APIKeyPoolExhaustedError):
        api_manager.chat_completion(messages=[{"role": "user", "content": "career plan"}], max_tokens=8)

    status = api_manager.provider_status()
    assert [item["status"] for item in status] == ["exhausted", "exhausted"]


@patch("backend.core.api_manager.OpenAI")
def test_model_list_health_check_uses_redacted_keys(openai_factory, api_manager) -> None:
    """Model-list validation checks keys without token-generating chat calls."""
    clients = {
        "key-alpha-0001": SequencedOpenAIClient([mocked_response("unused")]),
        "key-bravo-0002": SequencedOpenAIClient([mocked_response("unused")]),
    }
    openai_factory.side_effect = lambda api_key, base_url: clients[api_key]

    results = api_manager.model_list_health_check("moonshot")

    assert [result.status for result in results] == ["available", "available"]
    assert all("key-alpha-0001" not in result.key for result in results)


def test_configuration_warning_for_worker_keys_without_account(tmp_path) -> None:
    """Workers AI label-style keys are reported when the required account ID is absent."""
    env_path = tmp_path / ".env"
    env_path.write_text("worker ai key 1: worker-secret\n", encoding="utf-8")
    values, labeled = load_application_environment(env_path)
    from backend.core.api_manager import APIManager

    manager = APIManager(provider_configs=[])
    manager._env_values = values
    manager._labeled_keys = labeled
    manager._provider_configs = {"cloudflare": manager._load_cloudflare_config()}

    assert manager.configuration_warnings() == ["Cloudflare Workers AI keys were detected but no account ID/base URL is configured."]
