"""Smoke test: one tiny model call through Microsoft Foundry.

Run from the repository root:

    uv run python scripts/smoke_llm.py

Reads these variables from .env (see .env.example):

    ANTHROPIC_FOUNDRY_API_KEY   required
    ANTHROPIC_FOUNDRY_RESOURCE  resource name, or set ANTHROPIC_FOUNDRY_BASE_URL instead
    MODEL                       Foundry deployment name; defaults to claude-fable-5-2

Values in .env take precedence over the shell environment. If the shell exports one
endpoint variable and .env sets the other, the shell one is dropped, because the SDK
rejects having both.

The SDK's Foundry client reads the key and endpoint from the environment itself,
so this script never handles or prints the key.
"""

from __future__ import annotations

import os
import sys
from urllib.parse import urlsplit

import anthropic
from anthropic import AnthropicFoundry
from dotenv import dotenv_values, load_dotenv

DEFAULT_MODEL = "claude-fable-5-2"
PROMPT = "Reply with the single word: pong"
ENV_FILE = ".env"
RESOURCE_VAR = "ANTHROPIC_FOUNDRY_RESOURCE"
BASE_URL_VAR = "ANTHROPIC_FOUNDRY_BASE_URL"
FOUNDRY_VARS = ("ANTHROPIC_FOUNDRY_API_KEY", RESOURCE_VAR, BASE_URL_VAR)


def _drop_empty_env(names: tuple[str, ...]) -> None:
    """Remove variables set to an empty string, as a blank `KEY=` line in .env produces.

    The SDK treats an empty string as a provided value, which turns a blank line into a
    confusing error such as "base_url and resource are mutually exclusive".
    """
    for name in names:
        if name in os.environ and os.environ[name].strip() == "":
            del os.environ[name]


def _resolve_endpoint_conflict() -> None:
    """Keep the endpoint variable that .env defines when the shell exports the other one."""
    resource_set = bool(os.environ.get(RESOURCE_VAR))
    base_url_set = bool(os.environ.get(BASE_URL_VAR))
    if not (resource_set and base_url_set):
        return
    in_dotenv = {k for k, v in dotenv_values(ENV_FILE).items() if v and v.strip()}
    from_dotenv = {RESOURCE_VAR, BASE_URL_VAR} & in_dotenv
    if len(from_dotenv) != 1:
        print(
            f"set only one of {RESOURCE_VAR} and {BASE_URL_VAR}, in {ENV_FILE} or the shell",
            file=sys.stderr,
        )
        sys.exit(1)
    keep = from_dotenv.pop()
    drop = BASE_URL_VAR if keep == RESOURCE_VAR else RESOURCE_VAR
    del os.environ[drop]
    print(f"note: ignoring shell {drop}; using {keep} from {ENV_FILE}", file=sys.stderr)


def _check_base_url_path() -> None:
    """Fail early when the base URL path is not exactly /anthropic/.

    The portal shows a project endpoint (/api/projects/<name>/anthropic/) that the SDK
    cannot use. Any other path makes Azure answer "Missing required query parameter:
    api-version", which does not name the real cause.
    """
    base_url = os.environ.get(BASE_URL_VAR)
    if not base_url:
        return
    path = urlsplit(base_url).path.rstrip("/")
    if path == "/anthropic":
        return
    hint = ""
    if path.startswith("/api/projects/"):
        hint = " (that is the project endpoint from the portal; drop the /api/projects/<name> part)"
    print(
        f"{BASE_URL_VAR} path must be exactly /anthropic/, got {path}/{hint}. "
        f"Use https://<resource>.services.ai.azure.com/anthropic/ "
        f"or set {RESOURCE_VAR}=<resource> instead",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> int:
    load_dotenv(override=True)
    _drop_empty_env(FOUNDRY_VARS)
    _resolve_endpoint_conflict()
    _check_base_url_path()
    model = os.environ.get("MODEL") or DEFAULT_MODEL

    try:
        client = AnthropicFoundry()
    except (anthropic.AnthropicError, ValueError) as exc:
        print(f"client setup failed: {exc}", file=sys.stderr)
        return 1

    try:
        response = client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": PROMPT}],
        )
    except anthropic.AuthenticationError:
        print("authentication failed: check ANTHROPIC_FOUNDRY_API_KEY", file=sys.stderr)
        return 1
    except anthropic.NotFoundError as exc:
        print(f"not found (deployment or endpoint): {exc.message}", file=sys.stderr)
        return 1
    except anthropic.APIStatusError as exc:
        print(f"api error {exc.status_code}: {exc.message}", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError as exc:
        print(f"connection error: check {RESOURCE_VAR} or {BASE_URL_VAR} ({exc})", file=sys.stderr)
        return 1

    if response.stop_reason == "refusal":
        print(f"model refused: {response.stop_details}", file=sys.stderr)
        return 1

    text = "".join(block.text for block in response.content if block.type == "text")
    print(f"model:         {response.model}")
    print(f"stop_reason:   {response.stop_reason}")
    print(f"reply:         {text.strip()}")
    print(f"input_tokens:  {response.usage.input_tokens}")
    print(f"output_tokens: {response.usage.output_tokens}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
