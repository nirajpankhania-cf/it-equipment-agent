import json
from pathlib import Path
from typing import Any

from equipment.observability import RequestLogger, RunLogger, new_id


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_new_id_has_optional_prefix_and_is_unique() -> None:
    first, second = new_id("req"), new_id("req")
    assert first.startswith("req-")
    assert first != second
    assert not new_id().startswith("-")


def test_default_run_id_gives_single_run_prefix_in_filename(tmp_path: Path) -> None:
    with RunLogger(logs_dir=tmp_path) as run:
        pass
    assert run.path.name == f"run-{run.run_id}.jsonl"
    assert not run.path.name.startswith("run-run-")


def test_run_logger_writes_one_json_line_per_event(tmp_path: Path) -> None:
    with RunLogger(run_id="r1", logs_dir=tmp_path / "logs") as run:
        run.log("started", answer=42)
        run.log("warned", level="WARNING", request_id="q1")

    assert run.path == tmp_path / "logs" / "run-r1.jsonl"
    lines = read(run.path)
    assert [line["event"] for line in lines] == ["started", "warned"]
    assert all(line["run_id"] == "r1" for line in lines)
    assert lines[0]["request_id"] is None
    assert lines[0]["answer"] == 42
    assert lines[0]["level"] == "INFO"
    assert lines[1]["request_id"] == "q1"
    assert lines[1]["level"] == "WARNING"
    assert "ts" in lines[0]


def test_request_logger_binds_request_and_trace_ids(tmp_path: Path) -> None:
    with RunLogger(run_id="r1", logs_dir=tmp_path) as run:
        logger = run.for_request("req-1")
        logger.log("a")
        logger.log("b", level="ERROR", detail="x")

    lines = read(run.path)
    assert all(line["run_id"] == "r1" for line in lines)
    assert all(line["request_id"] == "req-1" for line in lines)
    assert all(line["trace_id"] == "req-1" for line in lines)
    assert lines[1]["detail"] == "x"


def test_for_request_generates_an_id(tmp_path: Path) -> None:
    with RunLogger(logs_dir=tmp_path) as run:
        logger = run.for_request()
    assert isinstance(logger, RequestLogger)
    assert logger.request_id.startswith("req-")
    assert logger.run_id == run.run_id


def test_span_records_duration_parent_and_fields(tmp_path: Path) -> None:
    with RunLogger(run_id="r1", logs_dir=tmp_path) as run:
        logger = run.for_request("req-1")
        with logger.span("model_call", purpose="react") as parent:
            with logger.span("tool_call", parent_span_id=parent.span_id, tool="x") as child:
                child.fields["result"] = {"found": True}

    lines = read(run.path)
    assert [line["name"] for line in lines] == ["tool_call", "model_call"]
    child_line, parent_line = lines
    assert child_line["parent_span_id"] == parent_line["span_id"]
    assert child_line["result"] == {"found": True}
    assert child_line["tool"] == "x"
    assert parent_line["purpose"] == "react"
    assert parent_line["parent_span_id"] is None
    assert parent_line["duration_ms"] >= child_line["duration_ms"] >= 0
    assert child_line["request_id"] == "req-1"


def test_span_is_written_even_when_block_raises(tmp_path: Path) -> None:
    with RunLogger(run_id="r1", logs_dir=tmp_path) as run:
        logger = run.for_request("req-1")
        try:
            with logger.span("tool_call"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass

    assert read(run.path)[0]["name"] == "tool_call"


def test_non_json_values_are_stringified(tmp_path: Path) -> None:
    with RunLogger(run_id="r1", logs_dir=tmp_path) as run:
        run.log("x", path=tmp_path)
    assert read(run.path)[0]["path"] == str(tmp_path)
