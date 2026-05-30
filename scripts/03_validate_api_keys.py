from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.api_manager import APIKeyPoolExhaustedError, APIManager


def build_parser() -> argparse.ArgumentParser:
    """Build the API key validation CLI parser."""
    parser = argparse.ArgumentParser(description="Validate configured AI provider keys without printing secrets.")
    parser.add_argument("--provider", action="append", help="Provider to validate. Defaults to every configured provider.")
    parser.add_argument("--chat-ping", action="store_true", help="Run one minimal chat request through the rotation path.")
    return parser


def main() -> int:
    """Run safe key validation and print redacted results."""
    args = build_parser().parse_args()
    manager = APIManager()
    providers = args.provider or manager.configured_providers()
    if not providers:
        print("No provider keys are configured.")
        return 1
    exit_code = 0
    for warning in manager.configuration_warnings():
        print(f"configuration-warning: {warning}")
        exit_code = 1
    for provider in providers:
        try:
            results = manager.model_list_health_check(provider)
        except APIKeyPoolExhaustedError as exc:
            print(f"{provider}: unavailable - {exc}")
            exit_code = 1
            continue
        for result in results:
            print(f"{result.provider}: {result.key} {result.status} model={result.model} detail={result.detail}")
            if result.status not in {"available", "active"}:
                exit_code = 1
    if args.chat_ping:
        try:
            ping = manager.chat_ping_health_check()
            print(f"{ping.provider}: {ping.key} {ping.status} model={ping.model} detail={ping.detail}")
        except Exception as exc:
            print(f"chat-ping: unavailable - {exc}")
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
