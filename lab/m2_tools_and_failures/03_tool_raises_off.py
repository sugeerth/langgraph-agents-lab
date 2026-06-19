"""03 — Turning error handling OFF: the exception propagates (fail-fast).

CONCEPT: ``handle_tool_errors=False`` disables the catch. A raising tool now propagates its
exception out of the ToolNode and aborts the run. Sometimes that's what you want — a bug in
a tool should be loud, not silently swallowed and "explained" to the model.

aha: error handling is a deliberate choice — catching is the default, not a law.
"""

from __future__ import annotations

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode

from lab.common import banner, make_flaky_tool
from lab.common.fake_model import tool_call


def _node(handle):
    return ToolNode([make_flaky_tool(fail_times=999)], handle_tool_errors=handle)


def demo_broken():
    """With handling OFF the exception escapes — we re-raise it for the caller to see.

    (For a fail-fast policy this *is* the desired behavior; we call it "broken" only in
    contrast to the graceful catch.)"""
    _node(False).invoke({"messages": [tool_call("flaky_tool", {"payload": "x"})]})


def demo_fixed():
    """Turn handling back ON: the same failure becomes a recoverable ToolMessage."""
    res = _node(True).invoke({"messages": [tool_call("flaky_tool", {"payload": "x"})]})
    return res["messages"][-1]


def run_demo():
    banner("03 — tool raises: handle_tool_errors=False -> exception propagates")
    print("BROKEN (handle_tool_errors=False):")
    try:
        demo_broken()
        print("  -> did NOT raise (unexpected)")
    except Exception as exc:  # noqa: BLE001
        print(f"  -> propagated {type(exc).__name__}: {exc}")
    msg = demo_fixed()
    print(f"FIXED (handle_tool_errors=True): ToolMessage(status={msg.status!r}) {msg.content!r}")


if __name__ == "__main__":
    run_demo()
