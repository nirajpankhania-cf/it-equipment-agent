"""A scripted stand-in for the model. Tests must never call the real model."""

from __future__ import annotations

import copy
import itertools
from typing import Any

from anthropic.types import Message, TextBlock, ThinkingBlock, ToolUseBlock, Usage

FAKE_MODEL = "claude-fable-5-2"
_ids = itertools.count(1)


def _message(
    content: list[TextBlock | ThinkingBlock | ToolUseBlock],
    *,
    stop_reason: str,
    input_tokens: int,
    output_tokens: int,
) -> Message:
    return Message.model_construct(
        id=f"msg_fake_{next(_ids)}",
        type="message",
        role="assistant",
        model=FAKE_MODEL,
        content=content,
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def text_message(text: str, *, input_tokens: int = 100, output_tokens: int = 20) -> Message:
    return _message(
        [TextBlock(type="text", text=text)],
        stop_reason="end_turn",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def tool_uses_message(
    calls: list[tuple[str, dict[str, Any]]],
    *,
    thought: str | None = "I need to look this up.",
    progress_update: str | None = None,
    input_tokens: int = 100,
    output_tokens: int = 30,
) -> Message:
    """A turn with one or more tool calls, each with a unique tool_use id.

    ``progress_update`` mimics a between-tool-call note returned as a thinking block.
    """
    content: list[TextBlock | ThinkingBlock | ToolUseBlock] = []
    if progress_update is not None:
        content.append(ThinkingBlock(type="thinking", thinking=progress_update, signature="sig"))
    if thought:
        content.append(TextBlock(type="text", text=thought))
    for name, arguments in calls:
        content.append(
            ToolUseBlock(id=f"toolu_{next(_ids)}", type="tool_use", name=name, input=arguments)
        )
    return _message(
        content, stop_reason="tool_use", input_tokens=input_tokens, output_tokens=output_tokens
    )


def tool_use_message(
    name: str,
    arguments: dict[str, Any],
    *,
    thought: str | None = "I need to look this up.",
    input_tokens: int = 100,
    output_tokens: int = 30,
) -> Message:
    return tool_uses_message(
        [(name, arguments)], thought=thought, input_tokens=input_tokens, output_tokens=output_tokens
    )


def decision_message(
    decision: str,
    reason_code: str,
    draft_response: str,
    *,
    ticket_id: str | None = None,
    thought: str | None = "I have what I need to decide.",
) -> Message:
    return tool_use_message(
        "submit_decision",
        {
            "decision": decision,
            "reason_code": reason_code,
            "ticket_id": ticket_id,
            "draft_response": draft_response,
        },
        thought=thought,
    )


class FakeModel:
    """Returns scripted messages in order and records every request it received."""

    def __init__(self, responses: list[Message]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Message:
        self.calls.append(copy.deepcopy(kwargs))  # snapshot: the caller mutates messages
        if not self.responses:
            raise AssertionError("FakeModel has no scripted response left")
        return self.responses.pop(0)


def critique_message(
    *, accurate: bool, commits_beyond_data: bool, issues: list[str] | None = None
) -> Message:
    return tool_use_message(
        "submit_critique",
        {"accurate": accurate, "commits_beyond_data": commits_beyond_data, "issues": issues or []},
        thought=None,
    )


def revision_message(draft_response: str) -> Message:
    return tool_use_message("submit_revision", {"draft_response": draft_response}, thought=None)


CLEAN_CRITIQUE = critique_message(accurate=True, commits_beyond_data=False)
