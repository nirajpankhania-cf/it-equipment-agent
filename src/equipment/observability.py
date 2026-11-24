"""JSON-lines logging and lightweight tracing.

One file per run at ``logs/run-<run_id>.jsonl``. Every line carries ``run_id`` and
``request_id``. Span records use OpenTelemetry field names (``trace_id``, ``span_id``,
``parent_span_id``) so a later move to real OTel export changes only this writer.

Never log the API key, request headers, or the environment.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

DEFAULT_LOGS_DIR = Path("logs")


def new_id(prefix: str | None = None) -> str:
    stamp = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    return f"{prefix}-{stamp}" if prefix else stamp


class RunLogger:
    """Appends one JSON object per line to the run's log file."""

    def __init__(self, run_id: str | None = None, logs_dir: Path = DEFAULT_LOGS_DIR) -> None:
        self.run_id = run_id or new_id()
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.path = logs_dir / f"run-{self.run_id}.jsonl"
        self._file = self.path.open("a", encoding="utf-8")

    def log(
        self,
        event: str,
        *,
        level: str = "INFO",
        request_id: str | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": level,
            "run_id": self.run_id,
            "request_id": request_id,
            "event": event,
            **fields,
        }
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()
        return record

    def for_request(self, request_id: str | None = None) -> RequestLogger:
        return RequestLogger(self, request_id or new_id("req"))

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class Span:
    span_id: str
    name: str
    parent_span_id: str | None
    fields: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None


class RequestLogger:
    """A RunLogger bound to one request. ``trace_id`` equals the request ID."""

    def __init__(self, run: RunLogger, request_id: str) -> None:
        self.run = run
        self.request_id = request_id

    @property
    def run_id(self) -> str:
        return self.run.run_id

    def log(self, event: str, *, level: str = "INFO", **fields: Any) -> dict[str, Any]:
        return self.run.log(
            event,
            level=level,
            request_id=self.request_id,
            trace_id=self.request_id,
            **fields,
        )

    @contextmanager
    def span(
        self, name: str, *, parent_span_id: str | None = None, **fields: Any
    ) -> Iterator[Span]:
        """Time a block and write one span record when it ends, even on error."""
        span = Span(uuid.uuid4().hex[:16], name, parent_span_id, dict(fields))
        start = time.perf_counter()
        try:
            yield span
        finally:
            span.duration_ms = round((time.perf_counter() - start) * 1000, 1)
            self.log(
                "span",
                span_id=span.span_id,
                parent_span_id=span.parent_span_id,
                name=span.name,
                duration_ms=span.duration_ms,
                **span.fields,
            )
