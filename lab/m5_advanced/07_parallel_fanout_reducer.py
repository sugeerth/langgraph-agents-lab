"""Parallel fan-out — why concurrent writes to one key NEED a reducer.

CONCEPT: when one node fans out to several nodes, those nodes run in the SAME super-step,
concurrently. If two of them write the same state key and that key has NO reducer, LangGraph
can't decide whose value wins, so it raises ``InvalidUpdateError`` ("can receive only one
value per step"). Declare a reducer (e.g. ``Annotated[list, operator.add]``) and the two
writes are MERGED instead of colliding.

aha: parallel nodes run in one super-step — a reducer is the merge contract for shared keys.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.errors import InvalidUpdateError
from langgraph.graph import END, START, StateGraph

from lab.common import banner


# --- BROKEN: no reducer on the shared key -------------------------------------------------
class BrokenState(TypedDict):
    results: list  # plain list, last-write-wins -> two concurrent writes collide


# --- FIXED: a reducer that merges concurrent writes ---------------------------------------
class FixedState(TypedDict):
    results: Annotated[list, operator.add]  # operator.add concatenates the two lists


def _fan(_state):
    return {}  # the fan-out node itself writes nothing


def _worker_a(_state):
    return {"results": ["A"]}


def _worker_b(_state):
    return {"results": ["B"]}


def _build(state_type):
    g = StateGraph(state_type)
    g.add_node("fan", _fan)
    g.add_node("a", _worker_a)
    g.add_node("b", _worker_b)
    g.add_edge(START, "fan")
    g.add_edge("fan", "a")  # both edges leave "fan" -> a and b run in parallel
    g.add_edge("fan", "b")
    g.add_edge("a", END)
    g.add_edge("b", END)
    return g.compile()


def demo_broken():
    """Run the no-reducer graph; capture the InvalidUpdateError it raises."""
    graph = _build(BrokenState)
    try:
        graph.invoke({"results": []})
        return {"raised": False, "error": None}
    except InvalidUpdateError as e:
        return {"raised": True, "error": str(e)}


def demo_fixed():
    """Run the reducer graph; both parallel writes merge into one list."""
    graph = _build(FixedState)
    out = graph.invoke({"results": []})
    return out


def run_demo():
    banner("BROKEN: two parallel nodes write 'results' with no reducer")
    broken = demo_broken()
    print(f"  raised InvalidUpdateError? {broken['raised']}")
    if broken["error"]:
        print(f"  message: {broken['error'].splitlines()[0]}")

    banner("FIXED: Annotated[list, operator.add] merges the concurrent writes")
    fixed = demo_fixed()
    print(f"  results merged = {sorted(fixed['results'])}")

    return {"broken": broken, "fixed": fixed}


if __name__ == "__main__":
    run_demo()
