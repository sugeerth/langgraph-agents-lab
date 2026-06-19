"""02 — You control exactly what the model sees when a tool fails.

CONCEPT: ``handle_tool_errors`` accepts more than ``True``. Pass a STRING to feed the model
a fixed instruction, or a CALLABLE ``(exc) -> str`` to build the message dynamically from
the exception. Either way the failure is delivered as a ``ToolMessage(status="error")``.

aha: a good error string is itself a prompt — it steers the model's next move.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode

from lab.common import banner, make_flaky_tool
from lab.common.fake_model import tool_call


def _run_with_handler(handler):
    """Invoke a ToolNode whose only tool always fails, using the given error handler."""
    tn = ToolNode([make_flaky_tool(fail_times=999)], handle_tool_errors=handler)
    return tn.invoke({"messages": [tool_call("flaky_tool", {"payload": "x"})]})


def demo_broken():
    """The bare default: a generic stringified exception — unhelpful to the model."""
    res = _run_with_handler(True)
    return res["messages"][-1]


def demo_fixed():
    """Two crafted forms — a fixed instruction string and a dynamic callable.

    Returns (string_msg, callable_msg) so a test can assert on both."""
    string_msg = _run_with_handler("Retry with a smaller value.")["messages"][-1]

    # A callable receives the raised exception and returns the text the model will read.
    def make_msg(exc: Exception) -> str:
        return f"Tool '{type(exc).__name__}' failed: {exc}. Try a different approach."

    callable_msg = _run_with_handler(make_msg)["messages"][-1]
    return string_msg, callable_msg


def run_demo():
    banner("02 — tool raises: custom error messages (string AND callable)")
    print(f"BROKEN (default, generic): {demo_broken().content!r}")
    string_msg, callable_msg = demo_fixed()
    print(f"FIXED (string):   {string_msg.content!r}  status={string_msg.status}")
    print(f"FIXED (callable): {callable_msg.content!r}  status={callable_msg.status}")


if __name__ == "__main__":
    run_demo()
