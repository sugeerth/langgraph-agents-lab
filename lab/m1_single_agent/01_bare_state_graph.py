"""A graph with NO LLM at all — just a TypedDict state and two plain python nodes.

CONCEPT: an "agent" is really a STATE MACHINE. You compile a graph of nodes; each node
receives the whole state and returns a *partial* dict that LangGraph merges back in.
aha: the LLM is optional — the graph machinery is the same whether a node calls Claude
or just does arithmetic.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from lab.common import banner


class CountState(TypedDict):
    """Plain state: no reducers, so each returned key is last-write-wins on merge."""

    value: int
    trail: list  # we overwrite this whole list in node_b to show the merge


def node_a(state: CountState) -> dict:
    """Node A returns ONLY the keys it wants to change — a partial-state update."""
    # Notice we don't return 'trail'; unchanged keys are left as-is by the merge.
    return {"value": state["value"] + 1, "trail": ["a ran"]}


def node_b(state: CountState) -> dict:
    """Node B reads what A wrote (state is shared) and returns its own partial update."""
    return {"value": state["value"] * 10, "trail": state["trail"] + ["b ran"]}


def build_graph():
    """START -> a -> b -> END. The edges define the control flow; no LLM involved."""
    g = StateGraph(CountState)
    g.add_node("a", node_a)
    g.add_node("b", node_b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    return g.compile()


def run_demo() -> dict:
    graph = build_graph()
    banner("01 — a bare StateGraph is just a state machine (no LLM)")
    # value starts at 1 -> node_a makes it 2 -> node_b makes it 20.
    out = graph.invoke({"value": 1, "trail": []})
    print("  start value : 1")
    print("  final value :", out["value"], "(1 +1 -> 2, then *10 -> 20)")
    print("  trail       :", out["trail"])
    return out


if __name__ == "__main__":
    run_demo()
