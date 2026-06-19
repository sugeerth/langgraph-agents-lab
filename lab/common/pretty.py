"""Tiny printing helpers so demos read clearly without re-implementing formatting."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


def _one(m: BaseMessage) -> str:
    role = type(m).__name__.replace("Message", "").lower()
    if isinstance(m, AIMessage) and m.tool_calls:
        calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in m.tool_calls)
        body = f"[tool_call] {calls}" + (f"  | {m.content}" if m.content else "")
    elif isinstance(m, ToolMessage):
        body = f"[tool:{m.name}] {m.content}"
    else:
        body = str(m.content)
    return f"  {role:>8} | {body}"


def print_messages(state_or_messages: Any, *, title: str | None = None) -> None:
    """Print a transcript from a state dict or a raw list of messages."""
    msgs = (
        state_or_messages["messages"]
        if isinstance(state_or_messages, dict)
        else state_or_messages
    )
    if title:
        print(f"\n=== {title} ===")
    for m in msgs:
        print(_one(m))


def banner(text: str) -> None:
    print(f"\n{'-' * 70}\n{text}\n{'-' * 70}")
