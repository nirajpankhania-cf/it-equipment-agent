import json
from pathlib import Path
from typing import Any

import pytest

from equipment.llm import (
    MAX_TOKENS,
    BudgetExceededError,
    LLMClient,
    ModelPrice,
    Pricing,
    PricingError,
    load_environment,
)
from equipment.observability import RequestLogger, RunLogger
from tests.agent.fake_model import FakeModel, text_message, tool_use_message

PRICING = Pricing(
    models={"claude-fable-5-2": ModelPrice(8.0, 40.0), "cheap": ModelPrice(1.0, 2.0)},
    max_cost_per_run_usd=0.01,
)


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def logger(tmp_path: Path) -> RequestLogger:
    return RunLogger(run_id="r1", logs_dir=tmp_path).for_request("req-1")


class TestPricing:
    def test_cost_formula(self) -> None:
        assert ModelPrice(8.0, 40.0).cost(12_000, 1_500) == pytest.approx(0.156)
        assert ModelPrice(8.0, 40.0).cost(0, 0) == 0.0

    def test_load_bundled_file(self) -> None:
        pricing = Pricing.load(Path("pricing.toml"))
        assert pricing.for_model("claude-fable-5-2") == ModelPrice(8.0, 40.0)
        assert pricing.max_cost_per_run_usd > 0

    def test_load_custom_file(self, tmp_path: Path) -> None:
        path = tmp_path / "pricing.toml"
        path.write_text(
            '[models."m"]\ninput_per_million = 1.5\noutput_per_million = 3\n'
            "[budget]\nmax_cost_per_run_usd = 0.5\n"
        )
        pricing = Pricing.load(path)
        assert pricing.for_model("m") == ModelPrice(1.5, 3.0)
        assert pricing.max_cost_per_run_usd == 0.5

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(PricingError, match="nowhere.toml"):
            Pricing.load(tmp_path / "nowhere.toml")

    def test_missing_budget_section(self, tmp_path: Path) -> None:
        path = tmp_path / "pricing.toml"
        path.write_text('[models."m"]\ninput_per_million = 1\noutput_per_million = 2\n')
        with pytest.raises(PricingError, match="budget"):
            Pricing.load(path)

    def test_unknown_model(self) -> None:
        with pytest.raises(PricingError, match="no price for model 'other'"):
            PRICING.for_model("other")


class TestLLMClient:
    async def test_writes_cost_record_after_every_call(self, logger: RequestLogger) -> None:
        fake = FakeModel([text_message("hi", input_tokens=12_000, output_tokens=1_500)])
        client = LLMClient(
            logger,
            model="claude-fable-5-2",
            pricing=Pricing(PRICING.models, max_cost_per_run_usd=5.0),
            create=fake.create,
            environment="test",
        )

        response = await client.complete(
            purpose="react", step=1, system="sys", messages=[{"role": "user", "content": "x"}]
        )

        assert response.content[0].type == "text"
        records = [line for line in read(logger.run.path) if line["event"] == "model_call"]
        assert len(records) == 1
        record = records[0]
        assert record["run_id"] == "r1"
        assert record["request_id"] == "req-1"
        assert record["purpose"] == "react"
        assert record["step"] == 1
        assert record["model"] == "claude-fable-5-2"
        assert record["input_tokens"] == 12_000
        assert record["output_tokens"] == 1_500
        assert record["cost_usd"] == pytest.approx(0.156)
        assert record["run_cost_usd"] == pytest.approx(0.156)
        assert record["latency_ms"] >= 0
        assert record["stop_reason"] == "end_turn"
        assert record["environment"] == "test"
        assert client.calls == 1
        assert client.total_cost_usd == pytest.approx(0.156)

    async def test_passes_model_max_tokens_system_messages_and_tools(
        self, logger: RequestLogger
    ) -> None:
        fake = FakeModel([tool_use_message("get_employee_info", {"employee_id": "E001"})])
        client = LLMClient(logger, model="cheap", pricing=PRICING, create=fake.create)
        tools: list[Any] = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]

        await client.complete(
            purpose="react",
            step=2,
            system="sys",
            messages=[{"role": "user", "content": "x"}],
            tools=tools,
        )

        call = fake.calls[0]
        assert call["model"] == "cheap"
        assert call["max_tokens"] == MAX_TOKENS == 1024
        assert call["system"] == "sys"
        assert call["messages"] == [{"role": "user", "content": "x"}]
        assert call["tools"] == tools
        assert call["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
        assert call["thinking"] == {"type": "adaptive", "display": "updates"}
        assert call["extra_headers"] == {"anthropic-beta": "thinking-display-updates-2026-08-18"}

    async def test_omits_tools_when_none(self, logger: RequestLogger) -> None:
        fake = FakeModel([text_message("ok")])
        client = LLMClient(logger, model="cheap", pricing=PRICING, create=fake.create)
        await client.complete(purpose="reflection", step=1, system="s", messages=[])
        assert "tools" not in fake.calls[0]
        assert "tool_choice" not in fake.calls[0]

    async def test_budget_exceeded_raises_after_recording(self, logger: RequestLogger) -> None:
        fake = FakeModel(
            [
                text_message("a", input_tokens=500, output_tokens=100),
                text_message("b", input_tokens=500, output_tokens=100),
            ]
        )
        client = LLMClient(logger, model="claude-fable-5-2", pricing=PRICING, create=fake.create)

        await client.complete(purpose="react", step=1, system="s", messages=[])
        with pytest.raises(BudgetExceededError):
            await client.complete(purpose="react", step=2, system="s", messages=[])

        events = [line["event"] for line in read(logger.run.path)]
        assert events == ["model_call", "model_call", "budget_exceeded"]
        assert client.calls == 2

    def test_model_read_from_environment(
        self, logger: RequestLogger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MODEL", "cheap")
        client = LLMClient(logger, pricing=PRICING, create=FakeModel([]).create)
        assert client.model == "cheap"

    def test_unpriced_model_fails_at_construction(
        self, logger: RequestLogger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MODEL", "mystery")
        with pytest.raises(PricingError, match="mystery"):
            LLMClient(logger, pricing=PRICING, create=FakeModel([]).create)


class TestLoadEnvironment:
    @pytest.fixture(autouse=True)
    def clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            "ANTHROPIC_FOUNDRY_API_KEY",
            "ANTHROPIC_FOUNDRY_RESOURCE",
            "ANTHROPIC_FOUNDRY_BASE_URL",
        ):
            monkeypatch.delenv(name, raising=False)

    def test_blank_lines_are_dropped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("ANTHROPIC_FOUNDRY_API_KEY=k\nANTHROPIC_FOUNDRY_BASE_URL=\n")
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "shell-resource")

        load_environment(str(env_file))

        import os

        assert os.environ["ANTHROPIC_FOUNDRY_API_KEY"] == "k"
        assert "ANTHROPIC_FOUNDRY_BASE_URL" not in os.environ
        assert os.environ["ANTHROPIC_FOUNDRY_RESOURCE"] == "shell-resource"

    def test_dotenv_wins_over_shell_on_conflict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text(
            "ANTHROPIC_FOUNDRY_API_KEY=k\n"
            "ANTHROPIC_FOUNDRY_BASE_URL=https://r.services.ai.azure.com/anthropic/\n"
        )
        monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "shell-resource")

        load_environment(str(env_file))

        import os

        assert "ANTHROPIC_FOUNDRY_RESOURCE" not in os.environ
        assert os.environ["ANTHROPIC_FOUNDRY_BASE_URL"].endswith("/anthropic/")

    def test_wrong_base_url_path_is_rejected(self, tmp_path: Path) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text(
            "ANTHROPIC_FOUNDRY_API_KEY=k\n"
            "ANTHROPIC_FOUNDRY_BASE_URL=https://r.services.ai.azure.com/api/projects/p/anthropic/\n"
        )
        with pytest.raises(OSError, match="exactly /anthropic/"):
            load_environment(str(env_file))
