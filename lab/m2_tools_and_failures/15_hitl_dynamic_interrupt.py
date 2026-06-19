"""15 — Dynamic HITL: pause MID-node with interrupt(), approve or deny the action.

CONCEPT: ``interrupt_before`` (file 14) pauses at a graph BOUNDARY. ``interrupt(payload)``
pauses INSIDE a node — you can compute context, surface a question, and branch on the human's
reply. Resume with ``Command(resume=...)``; the node re-runs from the top and ``interrupt()``
returns the resumed value. Approving runs the destructive action; denying skips it.

aha: interrupt() is dynamic (pause mid-node on demand) vs static interrupt_before (a boundary).
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from lab.common import banner, dangerous_delete


class GateState(TypedDict):
    target: str
    approved: bool
    result: str


def _ask_human(state: GateState):
    """Pause mid-node to ask for approval; resume value decides the branch."""
    decision = interrupt({"question": f"Approve deletion of {state['target']!r}? (yes/no)"})
    return {"approved": decision == "yes"}


def _maybe_delete(state: GateState):
    """Run the destructive tool only if the human approved."""
    if state["approved"]:
        return {"result": dangerous_delete.invoke({"target": state["target"]})}
    return {"result": f"skipped deletion of {state['target']!r} (denied)"}


def build_graph():
    g = StateGraph(GateState)
    g.add_node("ask_human", _ask_human)
    g.add_node("maybe_delete", _maybe_delete)
    g.add_edge(START, "ask_human")
    g.add_edge("ask_human", "maybe_delete")
    g.add_edge("maybe_delete", END)
    return g.compile(checkpointer=MemorySaver())  # interrupt() requires a checkpointer


def _run(decision: str, thread_id: str):
    graph = build_graph()
    cfg = {"configurable": {"thread_id": thread_id}}
    graph.invoke({"target": "prod-db", "approved": False, "result": ""}, cfg)  # pauses at interrupt()
    return graph.invoke(Command(resume=decision), cfg)  # resume with the human's answer


def demo_broken():
    """Without a gate the delete just happens — here's the unsupervised path for contrast."""
    return {"result": dangerous_delete.invoke({"target": "prod-db"})}


def demo_fixed():
    """Approve vs deny take DIFFERENT branches; return both so a test can compare them."""
    return {"approved": _run("yes", "dyn-yes"), "denied": _run("no", "dyn-no")}


def run_demo():
    banner("15 — dynamic HITL: interrupt() mid-node, resume with Command(resume='yes'/'no')")
    print(f"BROKEN (no gate): {demo_broken()['result']!r}")
    res = demo_fixed()
    print(f"FIXED approve -> {res['approved']['result']!r}")
    print(f"FIXED  deny   -> {res['denied']['result']!r}")


if __name__ == "__main__":
    run_demo()
