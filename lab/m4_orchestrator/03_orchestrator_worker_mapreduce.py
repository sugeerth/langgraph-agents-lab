"""Orchestrator-worker map-reduce: planner -> N parallel workers -> synthesizer.

CONCEPT: when the amount of work isn't known until runtime (one task per input item), you
can't hard-wire edges. ``Send`` lets a node DYNAMICALLY fan out — it emits one
``Send("worker", payload)`` per item, and LangGraph runs that many worker copies in
parallel. Each worker writes to a list key whose reducer is ``operator.add``, so the
partial results concatenate. The synthesizer then reads the whole list (the "reduce").

aha: dynamic fan-out = ``Send`` (one per item, the MAP); the list reducer is the REDUCE.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from lab.common import banner


class MapReduceState(TypedDict):
    items: list[str]
    # operator.add concatenates each worker's partial list -> the reduce step.
    results: Annotated[list[str], operator.add]
    summary: str


class WorkerState(TypedDict):
    """The slice of state a single worker sees — just its one item."""

    item: str


def plan(state: MapReduceState):
    """Used as a CONDITIONAL EDGE from START: emit one Send per item (the fan-out)."""
    return [Send("worker", {"item": it}) for it in state["items"]]


def worker(state: WorkerState) -> dict:
    """Process one item. Returns a 1-element list; the reducer merges all of them."""
    return {"results": [f"processed:{state['item']}"]}


def synthesize(state: MapReduceState) -> dict:
    """The reduce: read every worker's result and produce a single summary."""
    return {"summary": f"combined {len(state['results'])} results"}


def build_graph():
    g = StateGraph(MapReduceState)
    g.add_node("worker", worker)
    g.add_node("synthesize", synthesize)
    # plan is the edge function itself — it returns Sends, so there is no "plan" node.
    g.add_conditional_edges(START, plan, ["worker"])
    g.add_edge("worker", "synthesize")
    g.add_edge("synthesize", END)
    return g.compile()


def run_demo():
    banner("03 — orchestrator-worker map-reduce (Send fan-out + list-reducer fan-in)")
    graph = build_graph()
    items = ["alpha", "beta", "gamma", "delta"]
    result = graph.invoke({"items": items, "results": [], "summary": ""})
    print(f"  inputs  ({len(items)}): {items}")
    print(f"  results ({len(result['results'])}): {sorted(result['results'])}")
    print(f"  summary: {result['summary']}")
    assert len(result["results"]) == len(items)  # one result per input item
    return result


if __name__ == "__main__":
    run_demo()
