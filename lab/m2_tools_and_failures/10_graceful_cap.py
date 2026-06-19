"""10 — Prefer a graceful cap over a crash.

CONCEPT: a runaway loop shouldn't take the whole request down. Two ways to cap it:
  (a) DIY — wrap the raw cyclic graph's ``.invoke`` in ``try/except GraphRecursionError`` and
      return a sensible partial answer.
  (b) BUILT-IN — ``create_react_agent`` already does this via a ``remaining_steps`` budget:
      with a model that always wants another tool call, it ends with a plain AIMessage
      "Sorry, need more steps to process this request." (no tool_calls) instead of crashing.

aha: prefer a graceful cap over a crash — and the prebuilt agent gives you one for free.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from lab.common import banner, calculator, get_model
from lab.common.fake_model import FakeChatModel, tool_call

# Reuse the same cyclic graph as file 09 via the loader-free path (build inline to stay self-contained).
from langgraph.graph import START, StateGraph

from lab.common import MessagesState


def _cyclic_graph():
    g = StateGraph(MessagesState)
    g.add_node("spin", lambda state: {"messages": [AIMessage("still thinking...")]})
    g.add_edge(START, "spin")
    g.add_edge("spin", "spin")
    return g.compile()


def demo_broken(limit: int = 6):
    """(a, broken) The raw loop with no guard raises GraphRecursionError."""
    try:
        _cyclic_graph().invoke({"messages": [HumanMessage("go")]}, config={"recursion_limit": limit})
        return None
    except GraphRecursionError as exc:
        return exc


def demo_fixed(limit: int = 6):
    """(a, fixed) Catch the error and return a partial answer instead of crashing."""
    try:
        _cyclic_graph().invoke({"messages": [HumanMessage("go")]}, config={"recursion_limit": limit})
        return {"status": "ok"}  # unreachable for this loop
    except GraphRecursionError:
        return {"status": "capped", "answer": "Stopped after the step budget; here's a partial result."}


def demo_react_graceful(limit: int = 8):
    """(b) create_react_agent stops GRACEFULLY — no exception, a plain final message."""
    looping_model = FakeChatModel(
        script=[],
        default=lambda msgs: tool_call("calculator", {"a": 1, "b": 1, "op": "add"}, id="loop"),
    )
    agent = create_react_agent(looping_model, [calculator])
    return agent.invoke({"messages": [HumanMessage("loop forever")]}, config={"recursion_limit": limit})


def run_demo():
    banner("10 — graceful cap: catch the error (a) OR use the prebuilt agent's built-in stop (b)")
    print(f"(a) BROKEN raw loop: raised {type(demo_broken()).__name__}")
    print(f"(a) FIXED  raw loop: {demo_fixed()}")
    out = demo_react_graceful()
    last = out["messages"][-1]
    print(f"(b) create_react_agent ended WITHOUT crashing: {last.content!r} (tool_calls={last.tool_calls})")


if __name__ == "__main__":
    run_demo()
