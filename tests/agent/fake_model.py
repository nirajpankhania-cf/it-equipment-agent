"""A scripted stand-in for the model. Tests must never call the real model."""

from __future__ import annotations

from typing import Any

from anthropic.types import Message, TextBlock, ToolUseBlock, Usage

FAKE_MODEL = "claude-fable-5-2"


def _message(
    content: list[TextBlock | ToolUseBlock],
    *,
    stop_reason: str,
    input_tokens: int,
    output_tokens: int,
) -> Message:
    return Message.model_construct(
        id="msg_fake",
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


def tool_use_message(
    name: str,
    arguments: dict[str, Any],
    *,
    thought: str | None = "I need to look this up.",
    input_tokens: int = 100,
    output_tokens: int = 30,
) -> Message:
    content: list[TextBlock | ToolUseBlock] = []
    if thought:
        content.append(TextBlock(type="text", text=thought))
    content.append(ToolUseBlock(id="toolu_fake", type="tool_use", name=name, input=arguments))
    return _message(
        content, stop_reason="tool_use", input_tokens=input_tokens, output_tokens=output_tokens
    )


class FakeModel:
    """Returns scripted messages in order and records every request it received."""

    def __init__(self, responses: list[Message]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("FakeModel has no scripted response left")
        return self.responses.pop(0)
