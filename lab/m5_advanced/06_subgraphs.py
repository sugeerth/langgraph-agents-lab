"""Subgraphs — a compiled graph used as a single node inside a bigger graph.

CONCEPT: a compiled LangGraph is itself a Runnable, so you can drop it in as a *node* of a
parent graph. The subgraph encapsulates its own multi-step workflow; the parent just sees
"a node". When parent and child share a state key, data flows through transparently.
Stream the parent with ``subgraphs=True`` to peek INSIDE the child — each event carries a
namespace tuple that is empty for parent-level work and non-empty inside the subgraph.

aha: subgraphs encapsulate sub-workflows — compose graphs the way you compose functions.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from lab.common import banner


class State(TypedDict):
    text: str


def _build_subgraph():
    """A tiny 2-step workflow we want to reuse as a black box."""

    def clean(state: State) -> State:
        return {"text": state["text"].strip()}

    def shout(state: State) -> State:
        return {"text": state["text"].upper() + "!"}

    g = StateGraph(State)
    g.add_node("clean", clean)
    g.add_node("shout", shout)
    g.add_edge(START, "clean")
    g.add_edge("clean", "shout")
    g.add_edge("shout", END)
    return g.compile()


def build_graph():
    """Parent graph that uses the compiled subgraph as one of its nodes."""
    subgraph = _build_subgraph()

    def intro(state: State) -> State:
        return {"text": "  hello from the parent  "}

    parent = StateGraph(State)
    parent.add_node("intro", intro)
    parent.add_node("processor", subgraph)  # <-- the compiled subgraph IS a node
    parent.add_edge(START, "intro")
    parent.add_edge("intro", "processor")
    parent.add_edge("processor", END)
    return parent.compile()


def run_demo():
    graph = build_graph()

    banner("Invoke the parent — the subgraph runs inside the 'processor' node")
    out = graph.invoke({"text": ""})
    print(f"  final text: {out['text']!r}")

    banner("Stream with subgraphs=True to observe events INSIDE the subgraph")
    saw_subgraph = False
    for namespace, update in graph.stream({"text": ""}, stream_mode="updates", subgraphs=True):
        where = "subgraph" if namespace else "parent"
        print(f"  [{where:>8}] ns={namespace} update={update}")
        if namespace:  # non-empty namespace => this event came from inside the subgraph
            saw_subgraph = True

    return {"output": out, "saw_subgraph": saw_subgraph}


if __name__ == "__main__":
    run_demo()
