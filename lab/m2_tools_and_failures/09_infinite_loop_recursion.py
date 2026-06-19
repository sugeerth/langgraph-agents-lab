"""09 — An infinite loop, and the recursion limit that stops it.

CONCEPT: a cyclic graph with no exit (a node that edges back to itself) would run forever.
LangGraph guards every run with a ``recursion_limit`` (default 25). When a RAW StateGraph
exceeds it, it raises ``GraphRecursionError`` — a hard stop, not a graceful one.

(Contrast file 10: ``create_react_agent`` does NOT raise; it stops gracefully. Teach both.)

aha: the recursion limit is your circuit breaker against runaway loops.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, StateGraph

from lab.common import MessagesState, banner


def build_graph():
    """A raw graph whose single node edges back to itself — it never reaches END."""
    g = StateGraph(MessagesState)
    g.add_node("spin", lambda state: {"messages": [AIMessage("still thinking...")]})
    g.add_edge(START, "spin")
    g.add_edge("spin", "spin")  # the cycle with no exit condition
    return g.compile()


def demo_broken(limit: int = 6):
    """Run the cyclic graph; it blows past ``limit`` and raises GraphRecursionError.

    Returns the raised exception so the test can assert on its type without try/except."""
    graph = build_graph()
    try:
        graph.invoke({"messages": [HumanMessage("go")]}, config={"recursion_limit": limit})
        return None  # should never happen
    except GraphRecursionError as exc:
        return exc


def run_demo():
    banner("09 — infinite loop: raw cyclic graph -> GraphRecursionError")
    exc = demo_broken(limit=6)
    print(f"BROKEN: raised {type(exc).__name__}: {str(exc)[:90]}...")
    print("FIX preview (file 10): catch it for a partial answer, or use create_react_agent's")
    print("                       built-in graceful 'need more steps' stop.")


if __name__ == "__main__":
    run_demo()
