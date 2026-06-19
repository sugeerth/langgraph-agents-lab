"""Shared lab infrastructure: model factory, fake model, tools, state, printing."""

from .fake_model import FakeChatModel, parallel_tool_calls, tool_call
from .model_factory import DEFAULT_MODEL, get_model, get_resilient_model, use_fake
from .pretty import banner, print_messages
from .state import MessagesState, SupervisorState
from .tools import (
    bad_schema_tool,
    calculator,
    dangerous_delete,
    make_flaky_tool,
    make_slow_tool,
    mock_web_search,
)

__all__ = [
    "FakeChatModel",
    "tool_call",
    "parallel_tool_calls",
    "get_model",
    "get_resilient_model",
    "use_fake",
    "DEFAULT_MODEL",
    "MessagesState",
    "SupervisorState",
    "print_messages",
    "banner",
    "calculator",
    "mock_web_search",
    "dangerous_delete",
    "bad_schema_tool",
    "make_flaky_tool",
    "make_slow_tool",
]
