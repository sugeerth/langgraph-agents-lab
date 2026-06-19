"""Reducers: the merge rule for a state key.

CONCEPT: when a key is Annotated with a reducer, repeated/concurrent writes are MERGED by
that function instead of overwritten. ``add_messages`` appends to a transcript;
``operator.add`` sums numbers / concatenates lists. A plain (un-annotated) key is
last-write-wins.
aha: a reducer is just the function LangGraph uses to combine the old value with each
node's update for that one key.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from lab.common import banner, print_messages


class ReducerState(TypedDict):
    messages: Annotated[list, add_messages]  # reducer: APPEND (de-dupes by id)
    total: Annotated[int, operator.add]  # reducer: SUM every write
    last_label: str  # NO reducer -> last write wins (overwritten)


def step_one(state: ReducerState) -> dict:
    return {
        "messages": [AIMessage("one")],
        "total": 1,  # reducer adds this to whatever total already is
        "last_label": "set-by-one",
    }


def step_two(state: ReducerState) -> dict:
    return {
        "messages": [AIMessage("two")],
        "total": 41,  # 1 + 41 == 42 via operator.add
        "last_label": "set-by-two",  # clobbers 'set-by-one'
    }


def build_graph():
    g = StateGraph(ReducerState)
    g.add_node("one", step_one)
    g.add_node("two", step_two)
    g.add_edge(START, "one")
    g.add_edge("one", "two")
    g.add_edge("two", END)
    return g.compile()


def run_demo() -> dict:
    graph = build_graph()
    banner("02 — a reducer is the merge rule for a state key")
    out = graph.invoke({"messages": [HumanMessage("start")], "total": 0, "last_label": "init"})
    print_messages(out, title="messages (add_messages APPENDED every write)")
    print("\n  total      :", out["total"], "(operator.add summed 0 + 1 + 41 == 42)")
    print("  last_label :", out["last_label"], "(no reducer -> last write wins)")
    return out


if __name__ == "__main__":
    run_demo()
