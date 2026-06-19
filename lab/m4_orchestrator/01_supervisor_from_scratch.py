"""Supervisor-from-scratch: one central node that ROUTES to workers and loops until done.

CONCEPT: the supervisor pattern. Instead of workers talking to each other, a single
"supervisor" node decides who acts next, hands control to that worker, and gets control
back when the worker finishes. The loop ends when the supervisor routes to END.

aha: a supervisor is just ONE node whose only job is routing — the workers are dumb,
the intelligence lives in the router.

Here the supervisor follows a fixed PLAN (researcher -> writer -> FINISH) so the run is
deterministic and guaranteed to terminate. We use ``Command(goto=...)`` so the supervisor
both updates state AND chooses the next hop in a single return value.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from lab.common import SupervisorState, banner, print_messages

# A deterministic plan the supervisor walks through. Each entry is "who acts next".
# In a real app a model would produce this routing decision (see file 02).
PLAN = ["researcher", "writer", "FINISH"]


def supervisor(state: SupervisorState) -> Command:
    """The router. Pick the next worker from the plan based on how many steps we've taken.

    ``step_count`` has an ``operator.add`` reducer, so each worker bumps it by 1 and the
    supervisor reads it back as a cursor into PLAN."""
    pick = PLAN[min(state["step_count"], len(PLAN) - 1)]
    if pick == "FINISH":
        return Command(goto=END, update={"next": "FINISH"})
    # Record the choice and jump straight to that worker node.
    return Command(goto=pick, update={"next": pick})


def researcher(state: SupervisorState) -> Command:
    """A worker: does its bit, appends a message, then hands control BACK to the supervisor."""
    return Command(
        goto="supervisor",
        update={"messages": [AIMessage(content="researcher: gathered 3 facts")], "step_count": 1},
    )


def writer(state: SupervisorState) -> Command:
    return Command(
        goto="supervisor",
        update={"messages": [AIMessage(content="writer: drafted the summary")], "step_count": 1},
    )


def build_graph():
    g = StateGraph(SupervisorState)
    g.add_node("supervisor", supervisor)
    g.add_node("researcher", researcher)
    g.add_node("writer", writer)
    g.add_edge(START, "supervisor")
    # No static edges out of supervisor/workers — Command(goto=...) carries the routing.
    return g.compile()


def run_demo():
    banner("01 — supervisor from scratch (central router loops over workers)")
    graph = build_graph()
    result = graph.invoke({"messages": [HumanMessage("write a brief")], "next": "", "step_count": 0})
    print_messages(result, title="transcript")
    print(f"\nworkers ran (step_count) = {result['step_count']}, final next = {result['next']!r}")
    return result


if __name__ == "__main__":
    run_demo()
