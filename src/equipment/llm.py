"""The only module that calls the model.

Every call goes through :meth:`LLMClient.complete`, which records tokens, latency, and
cost from ``pricing.toml`` as one log line. One call site means no call goes unrecorded.
"""

from __future__ import annotations

import os
import sys
import time
import tomllib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from anthropic import AsyncAnthropicFoundry
from anthropic.types import Message, MessageParam, ToolParam
from dotenv import dotenv_values, load_dotenv

from equipment.observability import RequestLogger

DEFAULT_MODEL = "claude-fable-5-2"
MAX_TOKENS = 1024
# Notes the model writes between tool calls come back as progress-update thinking blocks,
# empty under the default display. "updates" returns their text so the trace shows a
# Thought before every Action. Beta on every platform, Foundry included.
THINKING = {"type": "adaptive", "display": "updates"}
BETA_HEADERS = {"anthropic-beta": "thinking-display-updates-2026-08-18"}
DEFAULT_PRICING_PATH = Path("pricing.toml")

RESOURCE_VAR = "ANTHROPIC_FOUNDRY_RESOURCE"
BASE_URL_VAR = "ANTHROPIC_FOUNDRY_BASE_URL"
API_KEY_VAR = "ANTHROPIC_FOUNDRY_API_KEY"

CreateFn = Callable[..., Awaitable[Message]]


class PricingError(ValueError):
    """pricing.toml is missing, malformed, or has no entry for the model."""


class BudgetExceededError(RuntimeError):
    """The run's cumulative cost passed ``max_cost_per_run_usd``."""


def load_environment(env_file: str = ".env") -> None:
    """Load ``.env`` over the shell environment and tidy the Foundry variables.

    Blank ``KEY=`` lines are dropped because the SDK treats an empty string as set. If
    both endpoint variables end up set, the one defined in ``.env`` wins. A base URL whose
    path is not exactly ``/anthropic/`` is rejected with the fix spelled out.
    """
    load_dotenv(env_file, override=True)
    for name in (API_KEY_VAR, RESOURCE_VAR, BASE_URL_VAR):
        if name in os.environ and os.environ[name].strip() == "":
            del os.environ[name]

    if os.environ.get(RESOURCE_VAR) and os.environ.get(BASE_URL_VAR):
        in_dotenv = {k for k, v in dotenv_values(env_file).items() if v and v.strip()}
        from_dotenv = {RESOURCE_VAR, BASE_URL_VAR} & in_dotenv
        if len(from_dotenv) != 1:
            raise OSError(f"set only one of {RESOURCE_VAR} and {BASE_URL_VAR}")
        keep = from_dotenv.pop()
        drop = BASE_URL_VAR if keep == RESOURCE_VAR else RESOURCE_VAR
        del os.environ[drop]
        print(f"note: ignoring shell {drop}; using {keep} from {env_file}", file=sys.stderr)

    base_url = os.environ.get(BASE_URL_VAR)
    if base_url and urlsplit(base_url).path.rstrip("/") != "/anthropic":
        raise OSError(
            f"{BASE_URL_VAR} path must be exactly /anthropic/, for example "
            f"https://<resource>.services.ai.azure.com/anthropic/"
        )


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return round(
            input_tokens / 1_000_000 * self.input_per_million
            + output_tokens / 1_000_000 * self.output_per_million,
            6,
        )


@dataclass(frozen=True)
class Pricing:
    models: dict[str, ModelPrice]
    max_cost_per_run_usd: float

    @classmethod
    def load(cls, path: Path = DEFAULT_PRICING_PATH) -> Pricing:
        try:
            with path.open("rb") as f:
                raw = tomllib.load(f)
            models = {
                name: ModelPrice(float(p["input_per_million"]), float(p["output_per_million"]))
                for name, p in raw["models"].items()
            }
            return cls(models, float(raw["budget"]["max_cost_per_run_usd"]))
        except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError, ValueError) as exc:
            raise PricingError(f"{path}: {exc!r}") from exc

    def for_model(self, model: str) -> ModelPrice:
        try:
            return self.models[model]
        except KeyError:
            raise PricingError(
                f"no price for model '{model}' in pricing.toml; known: {sorted(self.models)}"
            ) from None


class LLMClient:
    """Wraps the Foundry client. Records a cost line after every call and enforces the budget."""

    def __init__(
        self,
        logger: RequestLogger,
        *,
        model: str | None = None,
        pricing: Pricing | None = None,
        create: CreateFn | None = None,
        environment: str | None = None,
    ) -> None:
        self.logger = logger
        self.model = model or os.environ.get("MODEL") or DEFAULT_MODEL
        self.pricing = pricing or Pricing.load()
        self.price = self.pricing.for_model(self.model)
        self.environment = environment or os.environ.get("EQUIPMENT_ENV", "local")
        self._create: CreateFn = create or AsyncAnthropicFoundry().messages.create
        self.calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0

    async def complete(
        self,
        *,
        purpose: str,
        step: int,
        system: str,
        messages: list[MessageParam],
        tools: list[ToolParam] | None = None,
    ) -> Message:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": messages,
            "thinking": THINKING,
            "extra_headers": BETA_HEADERS,
        }
        if tools:
            kwargs["tools"] = tools

        start = time.perf_counter()
        response = await self._create(**kwargs)
        latency_ms = round((time.perf_counter() - start) * 1000, 1)

        usage = response.usage
        cost = self.price.cost(usage.input_tokens, usage.output_tokens)
        self.calls += 1
        self.total_input_tokens += usage.input_tokens
        self.total_output_tokens += usage.output_tokens
        self.total_cost_usd = round(self.total_cost_usd + cost, 6)

        self.logger.log(
            "model_call",
            purpose=purpose,
            step=step,
            model=response.model,
            requested_model=self.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_input_tokens=usage.cache_read_input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens,
            cost_usd=cost,
            run_cost_usd=self.total_cost_usd,
            latency_ms=latency_ms,
            stop_reason=response.stop_reason,
            environment=self.environment,
        )
        if self.total_cost_usd > self.pricing.max_cost_per_run_usd:
            self.logger.log(
                "budget_exceeded",
                level="ERROR",
                run_cost_usd=self.total_cost_usd,
                budget_usd=self.pricing.max_cost_per_run_usd,
            )
            raise BudgetExceededError(
                f"run cost ${self.total_cost_usd:.4f} exceeds budget "
                f"${self.pricing.max_cost_per_run_usd:.2f}"
            )
        return response
