"""A small library of toy tools used across the lab.

Every tool has a pydantic-typed signature (``@tool`` infers an ``args_schema`` from the
type hints), which is what lets LangGraph validate the model's arguments automatically.
The "broken" tools (flaky / slow / dangerous / bad-schema) exist specifically to
demonstrate failure modes in module 2.
"""

from __future__ import annotations

import time

from langchain_core.tools import tool


@tool
def calculator(a: float, b: float, op: str) -> str:
    """Do basic arithmetic. op is one of: add, sub, mul, div."""
    ops = {
        "add": lambda: a + b,
        "sub": lambda: a - b,
        "mul": lambda: a * b,
        "div": lambda: a / b if b != 0 else float("inf"),
    }
    if op not in ops:
        raise ValueError(f"unknown op {op!r}; expected one of {sorted(ops)}")
    return str(ops[op]())


_SEARCH_INDEX = {
    "langgraph": "LangGraph is a library for building stateful, multi-actor LLM apps as graphs.",
    "react agent": "ReAct interleaves reasoning and tool calls in a loop until it can answer.",
    "checkpointer": "A checkpointer persists graph state per thread_id, enabling memory + resume.",
}


@tool
def mock_web_search(query: str) -> str:
    """Search the (fake) web for a short factual answer."""
    key = query.lower().strip()
    for k, v in _SEARCH_INDEX.items():
        if k in key:
            return v
    return f"No results for {query!r}. Try 'langgraph', 'react agent', or 'checkpointer'."


@tool
def dangerous_delete(target: str) -> str:
    """Permanently delete a resource. DESTRUCTIVE — should be gated behind human approval."""
    # In the lab this only records intent; it never touches the filesystem.
    return f"[deleted] {target} (irreversible)"


@tool
def bad_schema_tool(x: int) -> str:
    """Echo an integer. Used to demonstrate argument-validation failures when the model
    passes a non-integer."""
    return f"got integer {x}"


def make_flaky_tool(fail_times: int = 2, exc: type[Exception] = ValueError):
    """Build a fresh tool that raises ``exc`` on its first ``fail_times`` calls, then
    succeeds. Each call to this factory gets its own isolated counter (no global state),
    which keeps retry demos and their tests reproducible."""
    state = {"calls": 0}

    @tool
    def flaky_tool(payload: str) -> str:
        """Process a payload over a flaky connection that often fails transiently."""
        state["calls"] += 1
        if state["calls"] <= fail_times:
            raise exc(f"transient failure on attempt {state['calls']}")
        return f"processed {payload!r} after {state['calls']} attempt(s)"

    return flaky_tool


def make_slow_tool(delay: float = 5.0):
    """Build a tool that sleeps for ``delay`` seconds — stands in for a hanging call."""

    @tool
    def slow_tool(payload: str) -> str:
        """Call a slow downstream service."""
        time.sleep(delay)
        return f"slow result for {payload!r}"

    return slow_tool
